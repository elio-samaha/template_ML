# 📊 Pandas Manipulation Cheat Sheet

```python
import pandas as pd
import numpy as np

# Example dataset
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=6, freq="D"),
    "asset": ["A","A","B","B","A","B"],
    "price": [100, 102, 200, 198, 103, 201],
    "volume": [10, 12, 20, 18, 15, 25]
})
print(df)
```

---

## 1. Indexing & selection

```python
df.loc[0, "price"]        # value by row+col
df.iloc[0, 2]             # same but position-based
df.query("asset == 'A'")  # SQL-like query
df[df["price"] > 100]     # boolean mask
df.sort_values(["asset","date"], ascending=[True, False])
```

---

## 2. Creating new columns

```python
df["return"] = df["price"].pct_change()          # percent change
df["log_price"] = np.log(df["price"])            # log transform
df["asset_volume"] = df["asset"] + "_" + df["volume"].astype(str)
```

---

## 3. GroupBy aggregations

```python
# Mean price per asset
df.groupby("asset")["price"].mean()

# Multiple aggregations
df.groupby("asset").agg(
    mean_price=("price","mean"),
    vol=("volume","sum"),
    last_price=("price","last")
)
```

---

## 4. Pivot / unstack / melt

```python
# Pivot: wide table of prices by asset/date
pivot = df.pivot(index="date", columns="asset", values="price")
print(pivot)

# Melt: go back to long format
long = pivot.reset_index().melt(id_vars="date", value_name="price")
print(long)

# Unstack: like pivot but from multi-index
g = df.set_index(["asset","date"])["price"]
print(g.unstack())  # assets as rows, dates as columns
```

---

## 5. Rolling & expanding windows (time series)

```python
df = df.sort_values("date")
df["rolling_mean"] = df.groupby("asset")["price"].transform(lambda s: s.rolling(3, min_periods=1).mean())
df["expanding_max"] = df.groupby("asset")["price"].transform(lambda s: s.expanding().max())
```

---

## 6. Shift & lags (features for models)

```python
df["price_lag1"] = df.groupby("asset")["price"].shift(1)
df["return_lag1"] = df.groupby("asset")["return"].shift(1)
df["future_price"] = df.groupby("asset")["price"].shift(-1)  # "lead"
```

---

## 7. Resampling (time-indexed data)

```python
df_time = df.set_index("date")
df_time.resample("W")["price"].mean()    # weekly mean
df_time.resample("M").agg({"price":"last","volume":"sum"})
```

---

## 8. Merge / join / concat

```python
# Merge (SQL-style join)
meta = pd.DataFrame({"asset":["A","B"], "sector":["Tech","Energy"]})
df = df.merge(meta, on="asset", how="left")

# Concat (stack vertically or horizontally)
df1 = df.iloc[:3]; df2 = df.iloc[3:]
pd.concat([df1, df2], axis=0)  # rows
```

---

## 9. Crosstab / pivot_table (quick summaries)

```python
pd.crosstab(df["asset"], df["sector"], values=df["volume"], aggfunc="sum")
df.pivot_table(index="asset", columns="sector", values="volume", aggfunc="mean", fill_value=0)
```

---

## 10. Vector tricks

```python
# ravel / values for numpy fast ops
arr = df["price"].values.ravel()   # flatten series to 1D np array
mean = arr.mean()

# rank / cum*
df["rank"] = df.groupby("asset")["price"].rank()
df["cumsum_vol"] = df.groupby("asset")["volume"].cumsum()
```

---

## 11. Missing values

```python
df.isna().sum()
df["price"] = df["price"].fillna(method="ffill")  # forward fill
df.dropna(subset=["price"], inplace=True)         # drop missing
```

---

## 12. MultiIndex tricks (useful in trading data)

```python
df_mi = df.set_index(["asset","date"]).sort_index()
print(df_mi.loc["A"])      # all rows for asset A
print(df_mi.xs("2024-01-01", level="date"))  # all assets at that date
```

---
