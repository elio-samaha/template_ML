* data IO, relabeling, cleaning
* EDA (plots + tests): distributions, correlation, QQ, ACF/PACF, ADF stationarity, Granger, cointegration
* preprocessing: numeric/categorical pipelines, optional power/log transforms, polynomial features
* model zoo (regression): OLS, transformed/regularized variants, polynomial regression, SVR (nonlinear), RandomForest, XGBoost, PyTorch MLP, ARIMA/GARCH sketch
* model zoo (classification): logistic, SVC, RF, XGBoost, PyTorch MLP
* CV/tuning: Grid/RandomizedSearch, TimeSeriesSplit toggle
* diagnostics: residual plots, Jarque–Bera/Shapiro, Breusch–Pagan/White, DW/Ljung–Box, metrics
* post-processing + feature importance

> Pro tip: flip `IS_TIME_SERIES=True` if your data are time-ordered.

---

## 🧩 Cell 1 — Imports, config, helpers

```python
# ===== Imports & config =====
import os, warnings, sys, math, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="talk")

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Optional packages (use if available)
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Statsmodels for tests/TS diagnostics
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox, normal_ad
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Optional GARCH via arch package
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# ===== Global switches (tweak for your task) =====
TASK = "regression"         # or "classification"
TARGET = "y"                # name of target column in your CSV
IS_TIME_SERIES = False      # True -> use TimeSeriesSplit and keep order
DATETIME_COL = None         # e.g., "timestamp" if you have a time index
ID_COLS = []                # list of id cols to drop
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Utility
def rmse(y_true, y_pred): return math.sqrt(mean_squared_error(y_true, y_pred))
def print_title(s): 
    print("\n" + "="*len(s))
    print(s)
    print("="*len(s))
```

---

## 📥 Cell 2 — Load data, relabel, basic cleaning

```python
# ===== Load your CSV =====
CSV_PATH = "data.csv"  # <-- change me
df = pd.read_csv(CSV_PATH)

# Optional: parse datetime
if DATETIME_COL and DATETIME_COL in df.columns:
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors='coerce')
    df = df.sort_values(DATETIME_COL)

# Drop obvious IDs
for c in ID_COLS:
    if c in df.columns:
        df = df.drop(columns=[c])

# Quick sanity
print_title("Head / Info")
print(df.head(3))
print(df.info())

# Optional relabel for classification (set TASK="classification" first)
# Example: if target is numeric but you need a binary label (e.g., y > median)
if TASK == "classification":
    if not pd.api.types.is_bool_dtype(df[TARGET]) and df[TARGET].nunique() > 2:
        thresh = df[TARGET].median()
        df[TARGET] = (df[TARGET] > thresh).astype(int)
        print(f"[Relabel] Converted continuous target to binary via median threshold ({thresh:.4f}).")

# Handle obvious constant/duplicate columns
n_before = df.shape[1]
dups = df.columns[df.T.duplicated()].tolist()
if dups:
    df = df.loc[:, ~df.T.duplicated()]
    print(f"[Clean] Dropped duplicate columns: {dups}")
const_cols = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
if const_cols:
    df = df.drop(columns=const_cols)
    print(f"[Clean] Dropped constant columns: {const_cols}")

# Train/valid split indices (keep time order if TS)
if IS_TIME_SERIES:
    split_idx = int((1 - TEST_SIZE) * len(df))
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
else:
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET] if TASK=="classification" else None)

y_train = df_train[TARGET]
y_test  = df_test[TARGET]
X_train = df_train.drop(columns=[TARGET])
X_test  = df_test.drop(columns=[TARGET])

# Identify dtypes
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]
print_title("Columns")
print(f"Numeric ({len(num_cols)}): {num_cols[:10]}{'...' if len(num_cols)>10 else ''}")
print(f"Categorical ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols)>10 else ''}")
```

---

## 🔎 Cell 3 — EDA & statistical tests (run selectively)

```python
# ===== EDA: distributions & correlation =====
print_title("Basic Describe")
display(df_train.describe(include='all'))

print_title("Missingness (%)")
miss = df_train.isna().mean().sort_values(ascending=False) * 100
print(miss[miss>0].round(2))

# Histograms for numerics
_ = df_train[num_cols].hist(bins=30, figsize=(16, 10))

# Correlation heatmap (numerics)
if len(num_cols) > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(df_train[num_cols].corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap")
    plt.show()

# QQ plot & normality (target)
if TASK == "regression":
    plt.figure(figsize=(6,6))
    qqplot(y_train, line='s')
    plt.title("QQ plot of target")
    plt.show()
    jb_stat, jb_p, _, _ = jarque_bera(y_train)
    print(f"Jarque-Bera p={jb_p:.3g} (normality larger p ~ more normal)")

# Time series plots / ACF/PACF / ADF
if IS_TIME_SERIES and DATETIME_COL:
    plt.figure(figsize=(12,4))
    plt.plot(df_train[DATETIME_COL], y_train)
    plt.title("Target over time")
    plt.show()
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    plot_acf(y_train, ax=ax[0], lags=40)
    plot_pacf(y_train, ax=ax[1], lags=40, method='ywm')
    plt.show()
    adf_stat, adf_p, _, _, crit, _ = adfuller(y_train.dropna())
    print(f"ADF p={adf_p:.3g} (p<0.05 suggests stationarity)")

# Granger causality (pairwise small demo; expensive)
if IS_TIME_SERIES and len(num_cols) >= 2:
    print_title("Granger causality (demo on first two numerics, maxlag=3)")
    try:
        _ = grangercausalitytests(df_train[[num_cols[0], num_cols[1]]].dropna(), maxlag=3, verbose=True)
    except Exception as e:
        print("Granger error:", e)

# Cointegration (Engle-Granger on a pair)
if IS_TIME_SERIES and len(num_cols) >= 2:
    print_title("Cointegration test (first two numerics)")
    try:
        score, pval, _ = coint(df_train[num_cols[0]].values, df_train[num_cols[1]].values)
        print(f"coint p={pval:.3g} (p<0.05 suggests cointegration)")
    except Exception as e:
        print("Coint error:", e)
```

---

## 🧪 Cell 4 — Preprocessing pipelines (numeric/categorical + options)

```python
# ===== Feature engineering options =====
USE_POLY = False            # set True for polynomial basis (regression)
POLY_DEGREE = 2

USE_POWER = False           # Yeo–Johnson power transform (handles negatives)
USE_LOG1P = False           # log1p on numerics (regression-ish)

# Transformers
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
scaler = StandardScaler(with_mean=not IS_TIME_SERIES)  # leave_mean if rolling windows

num_steps = []
num_steps.append(("impute", num_imputer))

# Optional numeric transforms
if USE_LOG1P:
    num_steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))
if USE_POWER:
    num_steps.append(("power", PowerTransformer(method="yeo-johnson")))
num_steps.append(("scale", scaler))

if TASK == "regression" and USE_POLY:
    num_steps.append(("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)))

numeric_pipe = Pipeline(num_steps)

categorical_pipe = Pipeline(steps=[
    ("impute", cat_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ],
    remainder="drop"
)

# Helper to build full pipelines
def make_pipeline(estimator):
    return Pipeline(steps=[
        ("prep", preprocess),
        ("model", estimator)
    ])
```

---

## 📈 Cell 5 — Regression models (+ tuning, diagnostics)

```python
if TASK == "regression":
    print_title("REGRESSION MODELS")

    # --- A) Linear Regression (OLS) ---
    ols = make_pipeline(LinearRegression())
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)

    # --- B) Ridge/Lasso with CV (robust baseline) ---
    ridge = make_pipeline(RidgeCV(alphas=np.logspace(-6, 3, 20), cv=5))
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    lasso = make_pipeline(LassoCV(alphas=None, cv=5, random_state=RANDOM_STATE, max_iter=10000))
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    # --- C) Nonlinear: SVR (RBF) ---
    svr = make_pipeline(SVR(kernel="rbf"))
    param_svr = {"model__C":[0.1,1,10], "model__gamma":["scale","auto"]}
    svr_cv = GridSearchCV(svr, param_grid=param_svr, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    svr_cv.fit(X_train, y_train)
    y_pred_svr = svr_cv.predict(X_test)
    print(f"SVR best params: {svr_cv.best_params_}")

    # --- D) Random Forest ---
    rf = make_pipeline(RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                             random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # --- E) XGBoost (if available) ---
    if HAS_XGB:
        xgb = make_pipeline(XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                                         subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                                         objective="reg:squarederror", n_jobs=-1))
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
    else:
        y_pred_xgb = None
        print("[Info] xgboost not available; skipping.")

    # --- F) (Optional) PyTorch MLP ---
    if HAS_TORCH:
        class MLP(nn.Module):
            def __init__(self, in_dim, hidden=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, 1)
                )
            def forward(self, x): return self.net(x)
        # Build a numeric design matrix via preprocess
        Xtr = preprocess.fit_transform(X_train)
        Xte = preprocess.transform(X_test)
        ytr = y_train.values.astype(np.float32).reshape(-1,1)

        model = MLP(Xtr.shape[1], hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
        ytr_t = torch.from_numpy(ytr)
        for epoch in range(200):
            model.train()
            opt.zero_grad()
            pred = model(Xtr_t)
            loss = loss_fn(pred, ytr_t)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            y_pred_mlp = model(torch.from_numpy(Xte.astype(np.float32))).numpy().ravel()
    else:
        y_pred_mlp = None

    # --- G) (Optional) ARIMA/GARCH sketches for TS target (univariate) ---
    if IS_TIME_SERIES and len(num_cols) == 0:  # pure univariate target
        try:
            arima = ARIMA(y_train, order=(1,1,1)).fit()
            y_pred_arima = arima.forecast(steps=len(y_test))
        except Exception as e:
            y_pred_arima = None
            print("ARIMA error:", e)
        if HAS_ARCH:
            try:
                ret = pd.Series(y_train).diff().dropna()
                garch = arch_model(ret, vol='Garch', p=1, q=1).fit(disp="off")
                print(garch.summary())
            except Exception as e:
                print("GARCH error:", e)

    # --- Metrics helper ---
    def eval_reg(name, y_true, y_pred):
        if y_pred is None: 
            print(f"{name}: skipped")
            return
        mse = mean_squared_error(y_true, y_pred)
        print(f"{name:12s} | RMSE={math.sqrt(mse):.4f}  MAE={mean_absolute_error(y_true, y_pred):.4f}  R2={r2_score(y_true, y_pred):.4f}")
        return y_pred

    print_title("Test metrics")
    _ = eval_reg("OLS", y_test, y_pred_ols)
    _ = eval_reg("RidgeCV", y_test, y_pred_ridge)
    _ = eval_reg("LassoCV", y_test, y_pred_lasso)
    _ = eval_reg("SVR", y_test, y_pred_svr)
    _ = eval_reg("RandomForest", y_test, y_pred_rf)
    _ = eval_reg("XGBoost", y_test, y_pred_xgb)
    _ = eval_reg("TorchMLP", y_test, y_pred_mlp)

    # --- Residual diagnostics for your chosen model (example: RidgeCV) ---
    y_hat = y_pred_ridge
    resid = y_test - y_hat

    print_title("Residual diagnostics (RidgeCV example)")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(resid, kde=True, ax=ax[0]); ax[0].set_title("Residuals histogram")
    sm.qqplot(resid, line="s", ax=ax[1]); ax[1].set_title("Residuals QQ")
    plt.show()

    # Normality tests
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    ad_stat, ad_p = normal_ad(resid)
    print(f"Jarque–Bera p={jb_p:.3g} | Anderson–Darling p~{ad_p:.3g}")

    # Heteroskedasticity tests (need exog design X)
    X_design = sm.add_constant(preprocess.transform(X_test))
    bp = het_breuschpagan(resid, X_design)
    wh = het_white(resid, X_design)
    print(f"Breusch–Pagan p={bp[1]:.3g} | White p={wh[1]:.3g}")

    # Autocorrelation tests
    dw = sms.durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    print(f"Durbin–Watson={dw:.3f} | Ljung–Box(10) p={lb['lb_pvalue'].iloc[0]:.3g}")
```

---

## 🔤 Cell 6 — Classification models (+ metrics & curves)

```python
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
```

---

## 🧯 Cell 7 — Post-processing & interpretations

```python
# Feature importance (tree models)
if TASK == "regression":
    try:
        model_rf = rf.named_steps["model"]
        feat_names = (preprocess.get_feature_names_out())
        imp = pd.Series(model_rf.feature_importances_, index=feat_names).sort_values(ascending=False)[:20]
        plt.figure(figsize=(8,6)); sns.barplot(x=imp.values, y=imp.index); plt.title("RF feature importance"); plt.show()
    except Exception as e:
        print("Feature importance note:", e)

# Save artifacts quickly (optional)
import joblib
joblib.dump(preprocess, "preprocess.joblib")
# e.g., joblib.dump(ridge, "ridge_pipeline.joblib")
```

---

### How to adapt fast

* **Unknown dataset?** Set `TARGET`, flip `TASK`, set `DATETIME_COL` if present, flip `IS_TIME_SERIES`. Rerun Cell 2–7.
* **Linear vs transformed**: toggle `USE_POWER`/`USE_LOG1P`/`USE_POLY`. For target log, wrap model in `TransformedTargetRegressor`.
* **TS safety**: if ordered data, set `IS_TIME_SERIES=True` → uses holdout by time and you can swap in `TimeSeriesSplit` for CV:
  `tscv = TimeSeriesSplit(n_splits=5); GridSearchCV(..., cv=tscv, ...)`.
* **Diagnostics talk-track**: quote Jarque–Bera/Shapiro (normality), Breusch–Pagan/White (heteroskedasticity), DW/Ljung–Box (serial correlation). For violations: switch to robust errors, log/Power transform, add lags/diffs/AR terms, or use tree/boosting.

