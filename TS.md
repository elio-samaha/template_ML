## 1. Variance Inflation Factor (VIF)

### Intuition

VIF measures **how much the variance of a regression coefficient is inflated due to collinearity** among predictors.
If predictors are independent, each coefficient variance is “as small as it should be.”
If two predictors are almost linear combinations of each other, their coefficients explode (unstable estimates).

### Math

For a regression with predictors ( X_1, X_2, \dots, X_p ):

* Regress ( X_j ) on all the others:
  [
  X_j = \beta_0 + \sum_{k \neq j} \beta_k X_k + \epsilon_j
  ]
* Compute ( R_j^2 ) — the ( R^2 ) of this regression.
* Then
  [
  \text{VIF}_j = \frac{1}{1 - R_j^2}
  ]

If ( X_j ) can be almost perfectly predicted by the other features, ( R_j^2 \to 1 \Rightarrow VIF \to \infty ).

### Relation to coefficient variance

In OLS,
[
\mathrm{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}
]
Collinearity ⇒ ( X^T X ) nearly singular ⇒ variances blow up.
VIF quantifies this inflation relative to the uncorrelated case.

### Interpretation

| VIF | Rule of thumb                           |
| --- | --------------------------------------- |
| 1   | No multicollinearity                    |
| 1–5 | Moderate correlation (often acceptable) |
| >10 | Serious multicollinearity (unstable)    |

### Example

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

# simulate correlated data
rng = np.random.default_rng(0)
x1 = rng.normal(size=100)
x2 = 0.9*x1 + rng.normal(scale=0.1, size=100)
x3 = rng.normal(size=100)
X = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3})
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
pd.Series(vif, index=X.columns)
```

→ `x2` will have a VIF near 10+, while `x1` also high, `x3` near 1.

---

## 2. Condition Number of a Matrix

### Intuition

Condition number (κ) measures how **sensitive** a matrix solution is to input perturbations.
In regression, ( \hat{\beta} = (X^T X)^{-1} X^T y ).
If ( X^T X ) is ill-conditioned, tiny noise in ( y ) or ( X ) causes large swings in ( \hat{\beta} ).

### Math

For a square invertible matrix ( A ):
[
\kappa(A) = |A| \cdot |A^{-1}| = \frac{\sigma_{\max}}{\sigma_{\min}}
]
where ( \sigma_{\max}, \sigma_{\min} ) are the largest/smallest singular values.

* ( \kappa \approx 1 ): stable
* ( \kappa > 30 ): moderate ill-conditioning
* ( \kappa > 1000 ): serious numerical instability

### Example

```python
import numpy as np
X = np.array([[1,2],[2.001,4]])
np.linalg.cond(X)
```

→ Very high, because columns nearly collinear.

### Relation to VIF

* Both diagnose **collinearity**.
* VIF works feature-by-feature; condition number gives a **global** measure of the entire ( X ) matrix.
* Large condition number ⇒ some VIFs will be large.

---

## 3. Does matrix conditioning explain multicollinearity?

Yes — mathematically, multicollinearity *is* ill-conditioning of ( X^T X ):

* When predictors are correlated, ( X^T X ) becomes nearly singular.
* The smallest singular value ( \sigma_{\min} ) → 0 ⇒ condition number → ∞.
* So poor conditioning is the linear algebraic signature of multicollinearity.

---

## 4. Multivariate target correlation (advanced)

This part in your EDA computes **how each feature (or set of features) shares information with the target** beyond linear correlation.

### Mutual Information (MI)

Captures *nonlinear dependencies*.
[
I(X;Y) = \int \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} dx,dy
]
It’s always ≥ 0, zero only when ( X ) and ( Y ) are independent.

In sklearn:

```python
from sklearn.feature_selection import mutual_info_regression
X = df.select_dtypes(np.number).drop(columns='y')
y = df['y']
mi = pd.Series(mutual_info_regression(X, y, random_state=0), index=X.columns)
mi.sort_values(ascending=False).plot.barh(title="Mutual Information with target")
```

It gives you nonlinear relevance ranking.

You can also compute **partial correlation** or **canonical correlation analysis (CCA)** if you want to measure group–target correlation when features are grouped.

---

## 5. Time Series Diagnostics & Feature Engineering

If your dataset is time-dependent, here’s the standard toolkit.

```python
from statsmodels.tsa.stattools import adfuller, acf, pacf, grangercausalitytests, coint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
```

### a) Stationarity: ADF test

```python
adf_stat, adf_p, *_ = adfuller(df["y"])
print(f"ADF stat={adf_stat:.3f}, p={adf_p:.3f}")
```

If p < 0.05 → reject H0 (unit root) → series is stationary.

### b) Differencing to remove trend

```python
df["y_diff"] = df["y"].diff().dropna()
```

Then re-run ADF to confirm stationarity.

### c) ACF / PACF plots

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df["y"].dropna(), lags=30)
plot_pacf(df["y"].dropna(), lags=30)
```

They reveal lag dependencies.

### d) Shapiro–Wilk for Gaussianity

```python
from scipy.stats import shapiro
stat,p = shapiro(df["y"].dropna())
print("Gaussian" if p>0.05 else "Not Gaussian")
```

### e) Granger causality

Check if one time series helps predict another:

```python
grangercausalitytests(df[["feature1","feature2"]], maxlag=5)
```

### f) Cointegration

For two non-stationary but co-moving series (like two stocks):

```python
score, p, _ = coint(df["x1"], df["x2"])
print(f"Cointegration p={p:.3g}")
```

---

## 6. Feature importance and explainability

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import shap

rf = RandomForestRegressor().fit(X, y)
perm = permutation_importance(rf, X, y)
pd.Series(perm.importances_mean, index=X.columns).sort_values().plot.barh()

# SHAP (if tree model)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

---

## 7. Dimensionality reduction visualization (PCA / t-SNE / UMAP)

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

pca = PCA(n_components=2).fit_transform(X)
sns.scatterplot(x=pca[:,0], y=pca[:,1], hue=y)

tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=y)

umap_2d = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
sns.scatterplot(x=umap_2d[:,0], y=umap_2d[:,1], hue=y)
```

---

## 8. Clustering & group-level summaries

### Clusters

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
df["cluster"] = kmeans.labels_
sns.pairplot(df, hue="cluster")
```

### Groupby summaries

```python
df.groupby("cluster").agg(
    mean_y=("y","mean"),
    std_y=("y","std"),
    count=("y","size")
)
```

---

## 9. Anything else for time series

* **Rolling windows:**
  `df['y_roll_mean'] = df['y'].rolling(7).mean()`
* **Lag features:**
  `df['y_lag1'] = df['y'].shift(1)`
* **Exponential smoothing:**
  `df['y_ewm'] = df['y'].ewm(span=10).mean()`
* **Seasonal decomposition:**

  ```python
  from statsmodels.tsa.seasonal import seasonal_decompose
  res = seasonal_decompose(df.set_index("date")["y"], model="additive", period=12)
  res.plot()
  ```

---

## 10. Summary mental map

| Concept              | Measures                          | Library                       | Why                             |
| -------------------- | --------------------------------- | ----------------------------- | ------------------------------- |
| Multicollinearity    | VIF, Condition number             | `statsmodels`, `numpy.linalg` | Stability of coefficients       |
| Target relationships | Mutual info, correlations         | `sklearn.feature_selection`   | Nonlinear relevance             |
| Stationarity         | ADF, differencing                 | `statsmodels.tsa`             | Validity of ARIMA/OLS on series |
| Dependence           | ACF, PACF, Granger, Cointegration | `statsmodels.tsa`             | Lag structure, causality        |
| Explainability       | Permutation, SHAP                 | `sklearn`, `shap`             | Model interpretability          |
| Dimensionality       | PCA, t-SNE, UMAP                  | `sklearn`, `umap`             | Visual separability             |
| Unsupervised         | KMeans, DBSCAN                    | `sklearn.cluster`             | Regime discovery                |
| Group summaries      | `groupby().agg()`                 | pandas                        | Cohort-level insights           |

---

awesome — here’s a **Time-Series EDA & Diagnostics template** you can drop into your notebook. It’s modular (functions you can call), time-aware (no leakage), and covers everything you asked: ACF/PACF, ADF, Shapiro, Granger, differencing, cointegration, rolling features, seasonal decomposition, feature importance (permutation + SHAP if available), DR (PCA/t-SNE/UMAP), clustering, and groupby summaries.

> Assumptions: your DataFrame is `df` with a datetime column `date` and target `y`. Replace names as needed.

---

## 📦 Setup & helpers

```python
# ===== Imports =====
import numpy as np, pandas as pd, warnings, matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import shapiro

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ===== Utility: ensure datetime index & sorting =====
def prepare_ts(df, date_col="date", target="y"):
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).dropna(subset=[date_col])
    df = df.set_index(date_col)
    return df

# ===== Leak-safe TS split indices (single holdout) =====
def ts_holdout_index(n, test_size=0.2):
    n_test = int(np.ceil(test_size * n))
    train_idx = np.arange(0, n - n_test)
    test_idx  = np.arange(n - n_test, n)
    return train_idx, test_idx
```

---

## 🔎 Core TS diagnostics (ACF/PACF, ADF, Shapiro, differencing)

```python
def ts_basic_diagnostics(df, target="y", max_lag=40, seasonal_period=None, plot=True):
    """
    - Line plot
    - ACF/PACF
    - ADF stationarity
    - Shapiro normality (on residual-ish: y - rolling mean)
    - Optional seasonal decomposition
    - Suggest differencing if needed
    """
    s = df[target].astype(float).dropna()

    if plot:
        plt.figure(figsize=(12,4))
        plt.plot(s.index, s.values)
        plt.title(f"{target} over time"); plt.tight_layout(); plt.show()

        fig, ax = plt.subplots(1,2, figsize=(12,4))
        plot_acf(s, lags=max_lag, ax=ax[0])
        plot_pacf(s, lags=max_lag, ax=ax[1], method='ywm')
        fig.suptitle("ACF / PACF"); plt.tight_layout(); plt.show()

    # ADF stationarity
    adf_stat, adf_p, *_ = adfuller(s.dropna(), autolag="AIC")
    print(f"ADF: stat={adf_stat:.3f}, p={adf_p:.3g}  --> p<0.05 suggests stationarity")

    # Shapiro on de-trended series (rough): remove rolling mean to reduce trend
    detr = (s - s.rolling(window=10, min_periods=1).mean()).dropna()
    sh_stat, sh_p = shapiro(detr.sample(n=min(5000, len(detr))) if len(detr)>5000 else detr)
    print(f"Shapiro (on de-trended sample): stat={sh_stat:.3f}, p={sh_p:.3g}  --> p>0.05 ~ Gaussian")

    if seasonal_period:
        try:
            res = seasonal_decompose(s, model="additive", period=seasonal_period)
            res.plot(); plt.suptitle("Seasonal decomposition"); plt.show()
        except Exception as e:
            print("Seasonal decomposition skipped:", e)

    # Differencing hint
    if adf_p >= 0.05:
        print("Hint: Non-stationary (by ADF). Try differencing:")
        print("    df['y_diff'] = df['y'].diff()  # then rerun diagnostics")
```

---

## 🔁 Differencing utilities (trend/seasonality)

```python
def make_differences(df, target="y", lags=(1,), seasonal_period=None):
    """
    Create lag differences; optionally seasonal difference.
    """
    out = df.copy()
    for lag in lags:
        out[f"{target}_diff{lag}"] = out[target].diff(lag)
    if seasonal_period:
        out[f"{target}_diff_season{seasonal_period}"] = out[target].diff(seasonal_period)
    return out
```

---

## 🔗 Granger causality & Cointegration

```python
def run_granger(df, x, y, maxlag=8):
    """
    Test if x Granger-causes y (does lagged x help predict y?)
    df: datetime-indexed DataFrame
    """
    data = df[[y, x]].dropna()
    try:
        print(f"Granger causality: does {x} -> {y}? (maxlag={maxlag})")
        grangercausalitytests(data[[y, x]], maxlag=maxlag, verbose=True)
    except Exception as e:
        print("Granger test error:", e)

def run_cointegration(df, x1, x2):
    """
    Engle-Granger test: are x1 and x2 cointegrated?
    """
    series1 = df[x1].astype(float).dropna()
    series2 = df[x2].astype(float).dropna()
    common_idx = series1.index.intersection(series2.index)
    score, p, _ = coint(series1.loc[common_idx], series2.loc[common_idx])
    print(f"Cointegration {x1} ~ {x2}: p={p:.3g}  --> p<0.05 suggests cointegration")
```

---

## 📈 Rolling features, lags, and leakage-safe split

```python
def add_ts_features(df, target="y", lag_list=(1,5,10), roll_list=(5,10,20), ewm_span=None):
    out = df.copy()
    # Lags
    for L in lag_list:
        out[f"{target}_lag{L}"] = out[target].shift(L)
    # Rolling stats on target (or add for exogenous vars similarly)
    for w in roll_list:
        out[f"{target}_rollmean{w}"] = out[target].rolling(w).mean()
        out[f"{target}_rollstd{w}"]  = out[target].rolling(w).std()
    if ewm_span:
        out[f"{target}_ewm{ewm_span}"] = out[target].ewm(span=ewm_span, adjust=False).mean()
    return out
```

---

## 🧪 Feature–Target dependence (MI), Permutation Importance, SHAP

```python
def feature_importance_suite(model, X_train, y_train, X_test=None, y_test=None, topk=20):
    """
    - Mutual Information (nonlinear relevance)
    - Permutation importance on the provided model
    - SHAP (if tree model & shap available)
    """
    # Mutual Information
    mi = pd.Series(mutual_info_regression(X_train, y_train, random_state=0), index=X_train.columns)
    print("Top MI features:")
    display(mi.sort_values(ascending=False).head(topk))

    # Permutation importance (train or test)
    try:
        target_X, target_y = (X_test, y_test) if (X_test is not None and y_test is not None) else (X_train, y_train)
        r = permutation_importance(model, target_X, target_y, n_repeats=10, random_state=0, n_jobs=-1)
        pi = pd.Series(r.importances_mean, index=target_X.columns).sort_values(ascending=False).head(topk)
        plt.figure(figsize=(8,6)); sns.barplot(x=pi.values, y=pi.index); plt.title("Permutation importance"); plt.tight_layout(); plt.show()
    except Exception as e:
        print("Permutation importance skipped:", e)

    # SHAP
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, show=True)
        except Exception as e:
            print("SHAP skipped:", e)
```

---

## 🔻 Dimensionality reduction plots (PCA / t-SNE / UMAP)

```python
def dr_plots(X, y=None, title_prefix=""):
    # PCA
    pca = PCA(n_components=2, random_state=0).fit_transform(X)
    plt.figure(figsize=(6,5))
    if y is None:
        plt.scatter(pca[:,0], pca[:,1], s=12)
    else:
        sns.scatterplot(x=pca[:,0], y=pca[:,1], hue=y, s=18, palette="viridis")
    plt.title(f"{title_prefix}PCA(2)")
    plt.tight_layout(); plt.show()

    # t-SNE (slow on large n; consider subsample)
    tsne = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(X)
    plt.figure(figsize=(6,5))
    if y is None:
        plt.scatter(tsne[:,0], tsne[:,1], s=12)
    else:
        sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=y, s=18, palette="viridis")
    plt.title(f"{title_prefix}t-SNE(2)"); plt.tight_layout(); plt.show()

    # UMAP
    if HAS_UMAP:
        um = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
        plt.figure(figsize=(6,5))
        if y is None:
            plt.scatter(um[:,0], um[:,1], s=12)
        else:
            sns.scatterplot(x=um[:,0], y=um[:,1], hue=y, s=18, palette="viridis")
        plt.title(f"{title_prefix}UMAP(2)"); plt.tight_layout(); plt.show()
```

---

## 📌 Unsupervised clustering + group summaries

```python
from sklearn.cluster import KMeans

def cluster_and_summarize(X, df_context=None, y=None, k=3):
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    labels = km.labels_
    if df_context is not None:
        tmp = df_context.copy()
        tmp["cluster"] = labels
        g = tmp.groupby("cluster").agg(count=("cluster","size"))
        if y is not None and y in tmp.columns:
            g["y_mean"] = tmp.groupby("cluster")[y].mean()
            g["y_std"]  = tmp.groupby("cluster")[y].std()
        display(g)
    sns.countplot(x=labels)
    plt.title("Cluster sizes"); plt.show()
    return labels
```

---

## 🧮 Groupby summaries (e.g., by calendar or clusters)

```python
def time_groupby_summaries(df, target="y", freq="W"):
    """
    Resample by freq: 'D','W','M','Q'
    """
    agg = df[target].resample(freq).agg(["mean","std","min","max","count"])
    display(agg.tail(10))
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    ax.plot(agg.index, agg["mean"]); ax.fill_between(agg.index, agg["mean"]-agg["std"], agg["mean"]+agg["std"], alpha=0.2)
    ax.set_title(f"{target} resampled ({freq}) mean ± std"); plt.tight_layout(); plt.show()
```

---

## 🚀 Example usage (glue)

```python
# 0) Prepare
df_ts = prepare_ts(df, date_col="date", target="y")

# 1) Basic diagnostics
ts_basic_diagnostics(df_ts, target="y", max_lag=40, seasonal_period=None, plot=True)

# 2) Differencing if non-stationary
df_ts = make_differences(df_ts, target="y", lags=(1,), seasonal_period=None)

# 3) Add simple TS features (no leakage)
df_ts = add_ts_features(df_ts, target="y", lag_list=(1,5,10), roll_list=(5,10,20), ewm_span=10)

# 4) Resample summaries
time_groupby_summaries(df_ts, target="y", freq="W")

# 5) Granger & cointegration (pick two series you suspect interact)
# run_granger(df_ts, x="some_exogenous_series", y="y", maxlag=5)
# run_cointegration(df_ts, x1="x_series1", x2="x_series2")

# 6) Build a leak-safe train/test split for modeling
n = len(df_ts.dropna())
train_idx, test_idx = ts_holdout_index(n, test_size=0.2)
df_train = df_ts.iloc[train_idx].copy()
df_test  = df_ts.iloc[test_idx].copy()

# 7) Prepare matrices for feature importance (example with only numeric)
X_train = df_train.select_dtypes(np.number).drop(columns=["y"])
y_train = df_train["y"].values
X_test  = df_test.select_dtypes(np.number).drop(columns=["y"])
y_test  = df_test["y"].values

# 8) Fit a baseline model (example: RandomForest)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 9) Importance & explainability
feature_importance_suite(rf, X_train, y_train, X_test, y_test, topk=15)

# 10) DR plots
dr_plots(X_train, y_train, title_prefix="Train ")

# 11) Clustering (unsupervised regimes)
labels = cluster_and_summarize(X_train, df_context=df_train, y="y", k=3)
```

---

### Notes you can say in an interview (fast bullets)

* **Stationarity first:** I run ADF and difference/seasonal-difference if needed before modeling dependencies; I inspect ACF/PACF to understand lag structure.
* **Leakage control:** all rolling/lag features are created **using past only**, and splits are **time-ordered** via `TimeSeriesSplit` or tail holdouts.
* **Explainability:** I report **permutation importance** (model-agnostic) and **SHAP** for tree models to vet drivers.
* **Regime discovery:** quick **KMeans** or **UMAP+t-SNE** visuals can reveal market regimes and suggest interaction terms.
* **Cointegration/Granger:** I test for equilibrium relationships and lead-lag behavior to justify spread or causal features.

---

awesome — let’s build a **time-series modeling pipeline** for your dataset step by step. You write the code; I’ll review/correct. Below I give you a precise **spec** for each step (what to create, which APIs to use, and success checks). No full code—just enough guidance to make it smooth and leak-free.

---

# Step 0 — Preconditions (what you should already have)

* `df` with columns `date` (datetime-like) and target `y`.
* Sorted by `date` and set as index (or you keep `date` as a column but always sort by it).

**Success check:** `df.index.is_monotonic_increasing` (if using index) or `df.sort_values('date')` shows monotone dates.

---

# Step 1 — Create leak-safe features

## 1A) Lags & rolling stats for `y`

* Create lags: `y_lag1, y_lag5, y_lag10`.
* Rolling means/std: `y_rollmean5, y_rollmean10, y_rollstd5`.
* Optionally an EWM: `y_ewm10`.

**APIs to use:** `Series.shift`, `Series.rolling(window).mean()/std()`, `Series.ewm(span).mean()`.

**Rule:** compute *before* any splitting; then **drop rows with NA** created by lags/rolls.

**Success check:** No NA remains in features/target after you drop initial rows.

## 1B) Exogenous features (if any)

* If you have other numeric features, you can also lag/roll them similarly (but keep feature count reasonable).

**Tip:** Don’t create features using *future* values (no `.shift(-k)`).

---

# Step 2 — Train/test split by time (holdout)

* Use a **tail holdout**: last 20% (or last fixed number of timesteps) for test.
* Create `X_train, y_train, X_test, y_test` using only **numeric** feature columns (no `y`).

**APIs:** basic pandas slicing with indices; or compute index cut point.

**Success check:** `X_train.index.max() < X_test.index.min()` (strictly earlier).

---

# Step 3 — Preprocessing

* Use `ColumnTransformer` with:

  * Numeric: `SimpleImputer(strategy='median')` + `StandardScaler()`.
  * (If you still have categoricals: `OneHotEncoder(handle_unknown='ignore')` path. But likely all numeric now.)

* For **linear models**, scaling is important; for **tree/boosting**, optional but fine.

**Success check:** `preprocess.get_feature_names_out()` works and transformed shapes match expectations.

---

# Step 4 — Baselines (time-aware CV)

We’ll compare at least three model families:

### 4A) Regularized linear

* **RidgeCV** and **ElasticNetCV** pipelines:

  * Ridge alphas on logspace (e.g., `1e-6 ... 1e3`).
  * ElasticNet: grid `l1_ratio=[0.1, 0.5, 0.9]`; use CV.

**CV:** Use `TimeSeriesSplit(n_splits=5)` (no shuffling).

**Metric:** Negative RMSE (`scoring='neg_root_mean_squared_error'`) or MAE.

### 4B) Gradient boosted trees

* **XGBRegressor** (or LightGBM if allowed) with **early stopping**.

  * Create an inner **validation split from the training period’s tail** (e.g., last 10–20% of train) and pass `eval_set=[(X_val, y_val)]` with `early_stopping_rounds=150`.
  * Reasonable starting params: `n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`.

**Success check:** Model reports best iteration < `n_estimators`; test predictions have sensible scale.

### 4C) A simple time-series linear model

* **ElasticNet on differenced target**:

  * Create `y_diff = y - y_lag1` and predict `y_diff` using lag/roll features of `y` (all lagged!).
  * Reconstruct level predictions on test via `y_hat_t = y_{t-1} + \widehat{\Delta y_t}`.
  * Good to compare against non-differenced fits.

**Success check:** Reconstructed predictions align in scale with `y_test`.

---

# Step 5 — Evaluation (test set)

Compute:

* RMSE, MAE, MAPE (if target > 0), and (R^2).
* Plot: `y_test` vs `y_pred` line plot over time; scatter `y_pred` vs `y_test`; residuals over time.

**Bonus:** A rolling out-of-sample evaluation using `TimeSeriesSplit` (walk-forward) to estimate stability.

---

# Step 6 — Diagnostics

* **Residual diagnostics** (pick your best model on test):

  * Histogram + KDE.
  * **ACF** of residuals (should be low/autocorrelation removed).
  * **Ljung–Box** p-value on residuals (no significant autocorrelation).
  * **Breusch–Pagan** (heteroskedasticity check) using a linear proxy design matrix (`statsmodels` OLS on test design to get residuals).

* **Feature importance**

  * For tree model: permutation importance (on test) and, if available, SHAP summary.

* **Stability**

  * Learning curve with **TimeSeriesSplit** (train size vs CV RMSE).

---

# Step 7 — Simple ablations (what matters?)

* Drop groups of features (e.g., remove all roll features, or remove lags beyond 5) and re-evaluate quickly to see contribution.
* Helps you build a clean, minimal model.

---



