# ===== Extra models + deeper residual diagnostics (REGRESSION) =====
from sklearn.linear_model import ElasticNetCV, HuberRegressor, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import learning_curve, KFold

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif

if TASK == "regression":
    print_title("EXTRA REGRESSION MODELS")

    # ---- ElasticNet (L1 + L2) with CV ----
    enet = make_pipeline(ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=None,
                                      cv=5, random_state=RANDOM_STATE, max_iter=10000))
    enet.fit(X_train, y_train)
    y_pred_enet = enet.predict(X_test)

    # ---- Robust regression (Huber) ----
    # good when you suspect outliers; robust to heavy-tailed errors
    huber = make_pipeline(HuberRegressor(epsilon=1.35, alpha=1e-4))  # alpha is L2-like
    huber.fit(X_train, y_train)
    y_pred_huber = huber.predict(X_test)

    # ---- Gradient Boosting (tree-based, regularized via depth/learning_rate) ----
    gbr = make_pipeline(GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=RANDOM_STATE
    ))
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)

    # ---- Quantile Regression (pinball loss) -> median regression (q=0.5) ----
    # Useful when your loss function is asymmetric or you want robust median fit
    try:
        qreg = make_pipeline(QuantileRegressor(quantile=0.5, alpha=1.0, solver="highs"))
        qreg.fit(X_train, y_train)
        y_pred_qreg = qreg.predict(X_test)
    except Exception as e:
        y_pred_qreg = None
        print("QuantileRegressor skipped:", e)

    # ---- Stacking (simple blender of ridge + RF + SVR) ----
    base_estimators = [
        ("ridge", RidgeCV(alphas=np.logspace(-6,3,20), cv=5)),
        ("rf", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                     max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)),
        ("svr", SVR(kernel="rbf", C=1.0, gamma="scale"))
    ]
    stack = StackingRegressor(
        estimators=[("ridge", make_pipeline(RidgeCV(alphas=np.logspace(-6,3,20), cv=5))),
                    ("rf", make_pipeline(RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                               max_depth=None, random_state=RANDOM_STATE, n_jobs=-1))),
                    ("svr", make_pipeline(SVR(kernel="rbf", C=1.0, gamma="scale")))],
        final_estimator=LinearRegression(), passthrough=False, n_jobs=-1
    )
    stack.fit(X_train, y_train)
    y_pred_stack = stack.predict(X_test)

    # ---- Evaluate all new models ----
    print_title("Extra Models — Test metrics")
    _ = eval_reg("ElasticNetCV", y_test, y_pred_enet)
    _ = eval_reg("Huber", y_test, y_pred_huber)
    _ = eval_reg("GradBoost", y_test, y_pred_gbr)
    _ = eval_reg("Quantile(0.5)", y_test, y_pred_qreg)
    _ = eval_reg("Stacking", y_test, y_pred_stack)

    # ========= Residual diagnostics (pick a model to probe) =========
    # You can swap 'gbr' for any fitted pipeline: enet, huber, stack, ridge, rf, etc.
    probe_model = gbr
    y_hat = probe_model.predict(X_test)
    resid = y_test - y_hat
    fitted = pd.Series(y_hat, index=y_test.index)

    print_title("Residual plots")
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    sns.scatterplot(x=fitted, y=resid, ax=ax[0])
    ax[0].axhline(0, color="k", lw=1); ax[0].set_title("Residuals vs Fitted")
    sns.scatterplot(x=np.sqrt(np.abs(fitted)), y=np.sqrt(np.abs(resid)), ax=ax[1])
    ax[1].set_title("Scale–Location (√|resid| vs √|fitted|)")
    sm.qqplot(resid, line="s", ax=ax[2]); ax[2].set_title("QQ plot residuals")
    plt.show()

    # Normality tests
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    ad_stat, ad_p = normal_ad(resid)
    print(f"Jarque–Bera p={jb_p:.3g} | Anderson–Darling p~{ad_p:.3g}")

    # Heteroskedasticity (Breusch–Pagan / White) using a linear proxy design
    X_design = sm.add_constant(preprocess.transform(X_test))
    bp = het_breuschpagan(resid, X_design)
    wh = het_white(resid, X_design)
    print(f"Breusch–Pagan p={bp[1]:.3g} | White p={wh[1]:.3g}")

    # Autocorrelation tests
    dw = sms.durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    print(f"Durbin–Watson={dw:.3f} | Ljung–Box(10) p={lb['lb_pvalue'].iloc[0]:.3g}")

    # ========= Influence diagnostics (linear proxy) =========
    # For influence/hat/Cook's, fit a plain OLS to the *numeric design* as a proxy
    try:
        ols_proxy = sm.OLS(y_test.values, X_design).fit()
        infl = OLSInfluence(ols_proxy)
        hat = infl.hat_matrix_diag
        cooks = infl.cooks_distance[0]

        thr_hat = 2 * X_design.shape[1] / X_design.shape[0]  # rough leverage threshold
        top_idx = np.argsort(cooks)[-5:][::-1]

        print_title("Influence (top-5 by Cook's distance)")
        for i in top_idx:
            print(f"idx={y_test.index[i]}, cook={cooks[i]:.4g}, hat={hat[i]:.4g}, resid={resid.iloc[i]:.4g}")

        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].scatter(hat, resid); ax[0].axvline(thr_hat, color="r", ls="--")
        ax[0].set_title("Residual vs leverage"); ax[0].set_xlabel("hat"); ax[0].set_ylabel("resid")
        ax[1].stem(np.arange(len(cooks)), cooks, use_line_collection=True); ax[1].set_title("Cook's distance")
        plt.show()
    except Exception as e:
        print("Influence diagnostics skipped:", e)

    # ========= Multicollinearity (VIF on numeric features) =========
    try:
        # Compute VIF on standardized numeric design (avoid OHE explosion)
        X_num_only = numeric_pipe.fit_transform(X_train[num_cols])
        X_num_test = numeric_pipe.transform(X_test[num_cols])
        X_vif = sm.add_constant(X_num_test)
        vif = pd.Series([_vif(X_vif, i) for i in range(1, X_vif.shape[1])], index=num_cols)
        print_title("Top VIF (multicollinearity)")
        print(vif.sort_values(ascending=False).head(10).round(2))
    except Exception as e:
        print("VIF skipped:", e)

    # ========= Permutation importance (model-agnostic) =========
    try:
        r = permutation_importance(probe_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        feat_names = preprocess.get_feature_names_out()
        pi = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False)[:20]
        plt.figure(figsize=(8,6)); sns.barplot(x=pi.values, y=pi.index); plt.title("Permutation importance"); plt.show()
    except Exception as e:
        print("Permutation importance skipped:", e)

    # ========= Partial dependence (for a couple of top features) =========
    try:
        top2 = pi.index[:2].tolist()
        PartialDependenceDisplay.from_estimator(probe_model, X_test, features=top2, kind="average")
        plt.show()
    except Exception as e:
        pass

    # ========= Learning curve (diagnose bias/variance) =========
    print_title("Learning curve (GBR example)")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) if not IS_TIME_SERIES else TimeSeriesSplit(n_splits=5)
    train_sizes, train_scores, valid_scores = learning_curve(
        probe_model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), shuffle=not IS_TIME_SERIES, random_state=RANDOM_STATE if not IS_TIME_SERIES else None
    )
    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, -train_scores.mean(axis=1), marker="o", label="Train RMSE")
    plt.plot(train_sizes, -valid_scores.mean(axis=1), marker="o", label="CV RMSE")
    plt.xlabel("Training size"); plt.ylabel("RMSE"); plt.legend(); plt.title("Learning curve"); plt.show()
