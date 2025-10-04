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

# EDA_&_statistical_tests.py
import numpy as np
import pandas as pd

def class_balance_report(y):
    vc = pd.Series(y).value_counts(normalize=True).sort_index()
    print("Class balance (proportions):")
    print(vc.to_string())
    if (vc.min() < 0.2) and (len(vc) == 2):
        print("[Hint] Strong class imbalance detected -> use class_weight='balanced', tune threshold, or AUC/PR-AUC.")

