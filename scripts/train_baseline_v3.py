# train_baseline_v3.py
import os, json
import pandas as pd, numpy as np
from pathlib import Path
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

PROC = Path(r"D:\RR_Hackathon_2025\data\processed")
OUT = Path(r"D:\RR_Hackathon_2025\models")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC/"model_data.csv", low_memory=False)

# Remove name-derived columns from features (they create huge numeric noise)
name_cols = ['player_name_players','player_name_bow']
for c in name_cols:
    if c in df.columns:
        print("[INFO] Dropping name-derived column from features:", c)

# Build feature list (exclude identifiers and targets)
exclude = set(['player_id','player_name','sold_price','sold','price','log_price'])
exclude.update(name_cols)
feature_cols = [c for c in df.columns if c not in exclude]

# Create scaled regression target: price in lakhs then log1p
df['price_lakhs'] = df['price'].fillna(0) / 1e5
df['log_price_lakhs'] = np.log1p(df['price_lakhs'])

print("Total rows:", len(df))
print("Sold counts:\n", df['sold'].value_counts())
print("Regression rows (sold>0):", (df['sold']==1).sum())

# Build categorical maps and save them
cat_cols = [c for c in feature_cols if df[c].dtype == object]
encoding_maps = {}
for c in cat_cols:
    vals = df[c].fillna("NA").astype(str).unique().tolist()
    encoding_maps[c] = vals  # list preserves order; inference can map unseen -> len(vals)
# Save maps
with open(OUT/"cat_maps.json","w",encoding="utf8") as f:
    json.dump(encoding_maps, f, ensure_ascii=False, indent=2)
print("[INFO] Saved cat maps ->", OUT/"cat_maps.json")

# Factorize categorical columns in df for training (in-place)
for c in cat_cols:
    df[c] = df[c].fillna("NA").astype(str)
    df[c], _ = pd.factorize(df[c])

# Fill numeric NaNs
X_all = df[feature_cols].fillna(0)
y_clf = df['sold'].astype(int)
y_reg = df['log_price_lakhs']

# Save numeric means/std for scaling if needed
num_cols = [c for c in feature_cols if c not in cat_cols]
num_stats = {c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in num_cols}
with open(OUT/"num_stats.json","w") as f:
    json.dump(num_stats, f, indent=2)
print("[INFO] Saved numeric stats ->", OUT/"num_stats.json")

# Classification
neg = (y_clf==0).sum(); pos = (y_clf==1).sum()
scale_pos_weight = neg / (pos + 1e-9)
if pos >= 2:
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_clf, test_size=0.2, stratify=y_clf, random_state=42)
else:
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_clf, test_size=0.2, random_state=42)

clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                         objective='binary', random_state=42, scale_pos_weight=scale_pos_weight, verbosity=-1)
clf.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)])
y_pred = clf.predict_proba(X_val)[:,1]
print("Classification AUC (val):", roc_auc_score(y_val, y_pred))
joblib.dump(clf, OUT/"lgbm_sold_classifier_v3.pkl")
print("Saved classifier v3")

# Regression - train only on sold rows (non-zero price)
sold_mask = df['sold'] == 1
df_sold = df.loc[sold_mask].copy()
if len(df_sold) < 20:
    print("[WARN] Too few sold rows for regression:", len(df_sold))
else:
    Xr = df_sold[feature_cols].fillna(0)
    yr = df_sold['log_price_lakhs']
    Xr_tr, Xr_val, yr_tr, yr_val = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                            objective='regression', random_state=42, verbosity=-1)
    reg.fit(Xr_tr, yr_tr, eval_set=[(Xr_val, yr_val)], callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)])
    pr = reg.predict(Xr_val)
    mse_log = mean_squared_error(yr_val, pr); rmse_log = mse_log**0.5
    price_val = np.expm1(yr_val) * 1e5
    price_hat = np.expm1(pr) * 1e5
    mse_price = mean_squared_error(price_val, price_hat); rmse_price = mse_price**0.5
    print("Regression RMSE (log-price-lakhs, val):", rmse_log)
    print("Regression RMSE (price-Rs, val):", rmse_price)
    joblib.dump(reg, OUT/"lgbm_price_regressor_v3.pkl")
    print("Saved regressor v3")

print("Done.")
