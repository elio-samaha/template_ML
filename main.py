from Preprocessing_pipelines import get_preprocess
from Classification_models import make_logistic_cv, fit_xgb_classifier_es, evaluate_classifiers
from EDA_&_statistical_tests import class_balance_report

# assume X_train, X_test, y_train, y_test, num_cols, cat_cols, IS_TIME_SERIES defined
preprocess = get_preprocess(num_cols, cat_cols, scale_numeric=True, is_time_series=IS_TIME_SERIES)

class_balance_report(y_train)

# Logistic with L1/L2/EN
log_cv = make_logistic_cv(preprocess, penalty_grid=("l1","l2","elasticnet"))
log_cv.fit(X_train, y_train)
pred_log = log_cv.predict(X_test)
proba_log = log_cv.predict_proba(X_test)[:,1]

# XGB with early stopping
xgb, pred_xgb, proba_xgb = fit_xgb_classifier_es(preprocess, X_train, y_train, X_test, y_test, is_time_series=IS_TIME_SERIES)

# RF/SVC if you want:
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rf = Pipeline([("prep", preprocess), ("model", RandomForestClassifier(
    n_estimators=500, min_samples_leaf=2, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced_subsample"))])
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test); proba_rf = rf.predict_proba(X_test)[:,1]

svc = Pipeline([("prep", preprocess), ("model", SVC(kernel="rbf", probability=True, class_weight="balanced"))])
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test); proba_svc = svc.predict_proba(X_test)[:,1]

# Evaluate
results = {
    "LogisticCV": {"pred": pred_log, "proba": proba_log},
    "XGB-ES": {"pred": pred_xgb, "proba": proba_xgb} if xgb else {"pred": None},
    "RF": {"pred": pred_rf, "proba": proba_rf},
    "SVC": {"pred": pred_svc, "proba": proba_svc},
}
evaluate_classifiers(results, y_test)
