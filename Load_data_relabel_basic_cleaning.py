# ===== Load your CSV =====
CSV_PATH = "data.csv"  # <-- change me
df = pd.read_csv(CSV_PATH)

# Optional: parse datetime
if DATETIME_COL and DATETIME_COL in df.columns:
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors='coerce')
    df = df.sort_values(DATETIME_COL)

# Drop obvious IDs
for c in ID_COLS:
    if c in df.columns:
        df = df.drop(columns=[c])

# Quick sanity
print_title("Head / Info")
print(df.head(3))
print(df.info())

# Optional relabel for classification (set TASK="classification" first)
# Example: if target is numeric but you need a binary label (e.g., y > median)
if TASK == "classification":
    if not pd.api.types.is_bool_dtype(df[TARGET]) and df[TARGET].nunique() > 2:
        thresh = df[TARGET].median()
        df[TARGET] = (df[TARGET] > thresh).astype(int)
        print(f"[Relabel] Converted continuous target to binary via median threshold ({thresh:.4f}).")

# Handle obvious constant/duplicate columns
n_before = df.shape[1]
dups = df.columns[df.T.duplicated()].tolist()
if dups:
    df = df.loc[:, ~df.T.duplicated()]
    print(f"[Clean] Dropped duplicate columns: {dups}")
const_cols = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
if const_cols:
    df = df.drop(columns=const_cols)
    print(f"[Clean] Dropped constant columns: {const_cols}")

# Train/valid split indices (keep time order if TS)
if IS_TIME_SERIES:
    split_idx = int((1 - TEST_SIZE) * len(df))
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
else:
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET] if TASK=="classification" else None)

y_train = df_train[TARGET]
y_test  = df_test[TARGET]
X_train = df_train.drop(columns=[TARGET])
X_test  = df_test.drop(columns=[TARGET])

# Identify dtypes
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]
print_title("Columns")
print(f"Numeric ({len(num_cols)}): {num_cols[:10]}{'...' if len(num_cols)>10 else ''}")
print(f"Categorical ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols)>10 else ''}")
