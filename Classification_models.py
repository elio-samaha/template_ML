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
