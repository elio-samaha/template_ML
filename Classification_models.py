if TASK == "classification":
    print_title("CLASSIFICATION MODELS")

    # NOTE: scale matters for SVC/LogReg; handled in preprocess.
    logit = make_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    logit.fit(X_train, y_train)
    y_proba_logit = logit.predict_proba(X_test)[:,1]
    y_pred_logit = (y_proba_logit >= 0.5).astype(int)

    svc = make_pipeline(SVC(kernel="rbf", probability=True, class_weight="balanced"))
    param_svc = {"model__C":[0.1,1,10], "model__gamma":["scale","auto"]}
    svc_cv = GridSearchCV(svc, param_grid=param_svc, scoring="roc_auc", cv=3, n_jobs=-1)
    svc_cv.fit(X_train, y_train)
    y_proba_svc = svc_cv.predict_proba(X_test)[:,1]
    y_pred_svc = (y_proba_svc >= 0.5).astype(int)
    print(f"SVC best params: {svc_cv.best_params_}")

    rfc = make_pipeline(RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                               random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"))
    rfc.fit(X_train, y_train)
    y_proba_rfc = rfc.predict_proba(X_test)[:,1]
    y_pred_rfc = (y_proba_rfc >= 0.5).astype(int)

    if HAS_XGB:
        xgbc = make_pipeline(XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5,
                                           subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                                           eval_metric="logloss", tree_method="hist", n_jobs=-1))
        xgbc.fit(X_train, y_train)
        y_proba_xgb = xgbc.predict_proba(X_test)[:,1]
        y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)
    else:
        y_pred_xgb = y_proba_xgb = None
        print("[Info] xgboost not available; skipping.")

    # Optional Torch MLP classifier
    if HAS_TORCH:
        class MLPc(nn.Module):
            def __init__(self, in_dim, hidden=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, 1), nn.Sigmoid()
                )
            def forward(self, x): return self.net(x)
        Xtr = preprocess.fit_transform(X_train); Xte = preprocess.transform(X_test)
        ytr = y_train.values.astype(np.float32).reshape(-1,1)
        model = MLPc(Xtr.shape[1], hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.BCELoss()
        Xtr_t = torch.from_numpy(Xtr.astype(np.float32)); ytr_t = torch.from_numpy(ytr)
        for epoch in range(200):
            model.train(); opt.zero_grad()
            pred = model(Xtr_t); loss = loss_fn(pred, ytr_t); loss.backward(); opt.step()
        with torch.no_grad():
            y_proba_mlp = model(torch.from_numpy(Xte.astype(np.float32))).numpy().ravel()
            y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)
    else:
        y_pred_mlp = y_proba_mlp = None

    # ---- Metrics ----
    def eval_cls(name, y_true, y_pred, y_proba=None):
        if y_pred is None: 
            print(f"{name}: skipped"); return
        acc = accuracy_score(y_true, y_pred)
        p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        s = f"Acc={acc:.3f}  P/R/F1={p:.3f}/{r:.3f}/{f:.3f}"
        if y_proba is not None and len(np.unique(y_true))==2:
            auc = roc_auc_score(y_true, y_proba)
            s += f"  ROC-AUC={auc:.3f}"
        print(f"{name:12s} | {s}")

    print_title("Test metrics")
    eval_cls("LogReg", y_test, y_pred_logit, y_proba_logit)
    eval_cls("SVC", y_test, y_pred_svc, y_proba_svc)
    eval_cls("RF", y_test, y_pred_rfc, y_proba_rfc)
    eval_cls("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
    eval_cls("TorchMLP", y_test, y_pred_mlp, y_proba_mlp)

    # Curves & confusion matrix (LogReg example)
    RocCurveDisplay.from_predictions(y_test, y_proba_logit)
    plt.show()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_logit, normalize="true")
    plt.show()

# Classification_models.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_fscore_support,
                             RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from Preprocessing_pipelines import get_preprocess
from Imports_config_helpers import make_val_split_idx

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ---------- evaluation ----------
def evaluate_classifiers(results, y_test):
    """
    results: dict name -> dict(pred=, proba=)
    """
    for name, out in results.items():
        y_pred = out.get("pred", None)
        y_proba = out.get("proba", None)
        if y_pred is None:
            print(f"{name}: skipped"); continue
        acc = accuracy_score(y_test, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        msg = f"{name:12s} | Acc={acc:.3f}  P/R/F1={p:.3f}/{r:.3f}/{f:.3f}"
        if y_proba is not None and len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba)
            msg += f"  ROC-AUC={auc:.3f}"
        print(msg)
    # Plot for the first available model with proba
    for name, out in results.items():
        if out.get("proba", None) is not None:
            RocCurveDisplay.from_predictions(y_test, out["proba"])
            ConfusionMatrixDisplay.from_predictions(y_test, out["pred"], normalize="true")
            break

# ---------- Logistic with L1/L2/ElasticNet via saga ----------
def make_logistic_cv(preprocess, penalty_grid=("l1","l2","elasticnet"),
                     C_grid=np.logspace(-3, 2, 8), l1_ratio_grid=(0.2, 0.5, 0.8),
                     max_iter=2000, class_weight="balanced", n_jobs=-1, cv=5):
    """
    Returns a GridSearchCV over logistic penalties using solver='saga' (supports all penalties).
    """
    base = Pipeline(steps=[
        ("prep", preprocess),
        ("model", LogisticRegression(
            solver="saga", penalty="l2", max_iter=max_iter, class_weight=class_weight, n_jobs=n_jobs
        ))
    ])

    param_grid = [
        {"model__penalty": ["l1"], "model__C": C_grid},  # l1 uses l1_ratio ignored
        {"model__penalty": ["l2"], "model__C": C_grid},
        {"model__penalty": ["elasticnet"], "model__C": C_grid, "model__l1_ratio": l1_ratio_grid},
    ]
    # restrict penalties if caller passed a subset
    param_grid = [d for d in param_grid if d["model__penalty"][0] in penalty_grid]

    clf = GridSearchCV(
        base, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=True
    )
    return clf

# ---------- XGBoost with early stopping ----------
def fit_xgb_classifier_es(preprocess, X_train, y_train, X_test, y_test,
                          is_time_series=False, valid_size=0.2, random_state=42):
    if not HAS_XGB:
        return None, None, None
    # Build design matrices
    X_all = pd.concat([X_train, X_test], axis=0)  # to reuse the same encoder
    preprocess.fit(X_all)  # fit OHE/scale on all to avoid unseen categories at test (or fit on train only if safer)
    Xtr = preprocess.transform(X_train)
    Xte = preprocess.transform(X_test)
    # carve train/valid for early stopping from TRAIN only
    train_idx, valid_idx = make_val_split_idx(Xtr.shape[0], test_size=valid_size, is_time_series=is_time_series, random_state=random_state)
    X_tr, y_tr = Xtr[train_idx], y_train.iloc[train_idx]
    X_val, y_val = Xtr[valid_idx], y_train.iloc[valid_idx]

    xgb = XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        reg_lambda=1.0, reg_alpha=0.0, random_state=random_state,
        objective="binary:logistic", tree_method="hist", n_jobs=-1
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        early_stopping_rounds=100,
        verbose=False
    )
    proba = xgb.predict_proba(Xte)[:,1]
    pred = (proba >= 0.5).astype(int)
    return xgb, pred, proba

# ---------- runnable example (plug into your main) ----------
if __name__ == "__main__":
    # Expect df_train/df_test, num_cols, cat_cols prepared elsewhere; here’s a sketch:
    # from Load_data_relabel_basic_cleaning import df_train, df_test, TARGET
    # num_cols = ...; cat_cols = ...
    pass

