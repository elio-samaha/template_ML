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

# Preprocessing_pipelines.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer

def get_preprocess(num_cols, cat_cols,
                   use_power=False, use_log1p=False,
                   scale_numeric=True, is_time_series=False,
                   poly_degree=None):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if use_log1p:
        num_steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))
    if use_power:
        num_steps.append(("power", PowerTransformer(method="yeo-johnson")))
    if scale_numeric:
        num_steps.append(("scale", StandardScaler(with_mean=not is_time_series)))
    if poly_degree is not None and poly_degree >= 2:
        from sklearn.preprocessing import PolynomialFeatures
        num_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))

    numeric_pipe = Pipeline(num_steps)
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocess

