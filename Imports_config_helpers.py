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

# Imports_config_helpers.py
import numpy as np

def make_val_split_idx(n_total, test_size=0.2, is_time_series=False, random_state=42):
    """
    Return train_idx, valid_idx for early stopping without leaking test data.
    If time series, uses a tail split; else random split.
    """
    n_valid = int(np.ceil(test_size * n_total))
    if is_time_series:
        train_idx = np.arange(0, n_total - n_valid)
        valid_idx = np.arange(n_total - n_valid, n_total)
    else:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n_total)
        valid_idx = perm[:n_valid]
        train_idx = perm[n_valid:]
    return train_idx, valid_idx
