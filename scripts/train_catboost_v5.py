#!/usr/bin/env python3
"""
train_catboost_v5.py
- Loads processed model_data.csv (has sold, price, etc.)
- Loads enhanced_players_features.csv and merges on player_id
- Builds feature_cols (excludes ids/targets and name columns)
- Trains CatBoostClassifier and CatBoostRegressor (log_price_lakhs target)
- Saves models & metadata to models/
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(r"D:\RR_Hackathon_2025")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# Files
MODEL_DATA = PROC / "model_data.csv"            # contains sold, price
ENHANCED = PROC / "enhanced_players_features.csv"  # new enhanced features

# Load
print("[INFO] Loading data...")
df_model = pd.read_csv(MODEL_DATA, low_memory=False)
df_enh = pd.read_csv(ENHANCED, low_memory=False)

# Normalize merge key
if 'player_id' not in df_model.columns or 'player_id' not in df_enh.columns:
    raise RuntimeError("player_id required in model_data.csv and enhanced_players_features.csv")

# Merge enhanced features into model_data (enhanced overwrites duplicates)
df = df_model.merge(df_enh, on='player_id', how='left', suffixes=('','_enh'))

# Ensure price/log target exists; create log_price_lakhs
df['price'] = df['price'].fillna(0)
df['price_lakhs'] = df['price'] / 1e5
df['log_price_lakhs'] = np.log1p(df['price_lakhs'])

# Build feature list
exclude = {'player_id','player_name','sold_price','sold','price','log_price','price_lakhs','log_price_lakhs'}
# remove name-derived columns if present
exclude.update({'player_name_players','player_name_bow','player_name_norm'})
feature_cols = [c for c in df.columns if c not in exclude]

print("[INFO] Total rows:", len(df))
print("[INFO] Candidate feature count:", len(feature_cols))

# Identify categorical features: object dtype OR low-cardinality (<50 unique)
cat_cols = []
for c in feature_cols:
    if df[c].dtype == object:
        cat_cols.append(c)
    else:
        # treat low-cardinality numerics as categorical too (IDs etc.)
        if df[c].nunique(dropna=False) < 50:
            if df[c].nunique() / max(1, len(df)) < 0.05:
                cat_cols.append(c)

# Remove columns that are obviously numeric stats from cat_cols
numeric_keep = {'runs','balls','sr','ipl_balls','ipl_runs_conceded','ipl_wickets','ipl_economy','ipl_sr'}
cat_cols = [c for c in cat_cols if c not in numeric_keep]

print("[INFO] Detected categorical columns:", cat_cols)

# Prepare data: fillna and type cast
X = df[feature_cols].copy()
for c in feature_cols:
    if c in cat_cols:
        X[c] = X[c].fillna("NA").astype(str)
    else:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

# Label targets
y_clf = df['sold'].fillna(0).astype(int)
y_reg = df['log_price_lakhs'].fillna(0.0)

# Train/val split for classifier
X_tr, X_val, y_tr, y_val = train_test_split(X, y_clf, test_size=0.2, stratify=y_clf, random_state=42)

# CatBoost pools (specify categorical feature names)
cat_features = [i for i, c in enumerate(X.columns) if c in cat_cols]

print("[INFO] Training CatBoostClassifier...")
clf = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100
)

pool_tr = Pool(X_tr, y_tr, cat_features=cat_features if len(cat_features)>0 else None)
pool_val = Pool(X_val, y_val, cat_features=cat_features if len(cat_features)>0 else None)
clf.fit(pool_tr, eval_set=pool_val)
y_pred = clf.predict_proba(pool_val)[:,1]
print("Classifier AUC (val):", roc_auc_score(y_val, y_pred))
clf.save_model(str(MODELS/"catboost_classifier_v5.cbm"))
print("[INFO] Saved classifier ->", MODELS/"catboost_classifier_v5.cbm")

# Regression: only on sold rows
sold_mask = (df['sold']==1)
if sold_mask.sum() < 20:
    print("[WARN] Too few sold rows for robust regression:", sold_mask.sum())
else:
    Xr = X[sold_mask].copy()
    yr = y_reg[sold_mask].copy()
    Xr_tr, Xr_val, yr_tr, yr_val = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    pool_r_tr = Pool(Xr_tr, yr_tr, cat_features=cat_features if len(cat_features)>0 else None)
    pool_r_val = Pool(Xr_val, yr_val, cat_features=cat_features if len(cat_features)>0 else None)

    print("[INFO] Training CatBoostRegressor...")
    reg = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100
    )
    reg.fit(pool_r_tr, eval_set=pool_r_val)
    pr = reg.predict(pool_r_val)

    # compute RMSE using numpy (avoid sklearn 'squared' kw compatibility issues)
    rmse_log = float(np.sqrt(np.mean((yr_val.values - pr) ** 2)))
    # back-transform price to rupees for interpretability
    price_val = np.expm1(yr_val.values) * 1e5
    price_hat = np.expm1(pr) * 1e5
    rmse_price = float(np.sqrt(np.mean((price_val - price_hat) ** 2)))

    print("Regression RMSE (log-price-lakhs, val):", rmse_log)
    print("Regression RMSE (price-Rs, val):", rmse_price)
    reg.save_model(str(MODELS/"catboost_regressor_v5.cbm"))
    print("[INFO] Saved regressor ->", MODELS/"catboost_regressor_v5.cbm")

# Save metadata (feature order, cat_cols)
meta = {
    "feature_cols": feature_cols,
    "cat_cols": cat_cols
}
with open(MODELS/"catboost_meta_v5.json","w",encoding="utf8") as f:
    json.dump(meta, f, indent=2)
print("[DONE] Saved metadata ->", MODELS/"catboost_meta_v5.json")
