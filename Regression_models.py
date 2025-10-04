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
