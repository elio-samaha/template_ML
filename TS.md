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

Would you like me to build a **Time Series EDA template** (code + explanation comments) that merges all those diagnostics into one ready-to-run section you can drop into your notebook next?
