# train_baseline_v2.py
"""
Backward-compatible training script for:
 - Classification: predict sold (binary)
 - Regression: predict log_price (trained on sold rows only)

Fixes included:
 - Uses LightGBM callback-based early stopping (works with older lgb versions)
 - Computes RMSE as sqrt(MSE) for sklearn versions that don't support squared=False
 - Simple categorical factorization and missing-value handling
 - Saves models using joblib
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

# Paths - adjust if needed
PROC_DIR = r"D:\RR_Hackathon_2025\data\processed"
OUT_DIR = r"D:\RR_Hackathon_2025\models"
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(PROC_DIR, "model_data.csv"), low_memory=False)

# Basic cleaning
df = df[~df['player_name'].isna()].copy()
print("Total rows:", len(df))
print("Sold counts:\n", df['sold'].value_counts())

# Exclude identifiers + targets
exclude = {'player_id','player_name','sold_price','sold','price','log_price'}
all_cols = [c for c in df.columns if c not in exclude]

# Factorize object columns
cat_cols = [c for c in all_cols if df[c].dtype == object]
for c in cat_cols:
    df[c] = df[c].fillna('NA').astype(str)
    df[c], _ = pd.factorize(df[c])

# Fill numeric NaNs with 0
X = df[all_cols].fillna(0)
y_clf = df['sold'].astype(int)
y_reg = df['log_price']

print("Features used (count={}): {}".format(len(all_cols), all_cols))

# -------------------------
# Classification (sold vs unsold)
# -------------------------
neg = (y_clf==0).sum()
pos = (y_clf==1).sum()
print("neg, pos:", neg, pos)
scale_pos_weight = neg / (pos + 1e-9)

# Stratified split if possible
if pos >= 2:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y_clf, test_size=0.2, stratify=y_clf, random_state=42)
else:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y_clf, test_size=0.2, random_state=42)

clf_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'objective': 'binary',
    'verbosity': -1,
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight
}

clf = lgb.LGBMClassifier(**clf_params)

# Use callback-based early stopping for compatibility
clf.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

# Predict & evaluate
y_pred_proba = clf.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, y_pred_proba)
print("Classification AUC (val):", auc)

clf_path = os.path.join(OUT_DIR, "lgbm_sold_classifier.pkl")
joblib.dump(clf, clf_path)
print("Saved classifier ->", clf_path)

# -------------------------
# Regression (log_price) - train only on sold rows
# -------------------------
sold_mask = df['sold'] == 1
if sold_mask.sum() < 20:
    print("Not enough sold rows for regression. Sold rows:", sold_mask.sum())
else:
    df_sold = df.loc[sold_mask].copy()
    # Cap extreme log_price at 99.5th percentile to reduce outlier effect
    q995 = df_sold['log_price'].quantile(0.995)
    df_sold['log_price_capped'] = df_sold['log_price'].clip(upper=q995)

    Xr = df_sold[all_cols].fillna(0)
    yr = df_sold['log_price_capped']

    Xr_tr, Xr_val, yr_tr, yr_val = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_params = {
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'regression',
        'verbosity': -1,
        'random_state': 42
    }
    reg = lgb.LGBMRegressor(**reg_params)

    reg.fit(
        Xr_tr, yr_tr,
        eval_set=[(Xr_val, yr_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )

    pr = reg.predict(Xr_val)

    # Backwards-compatible RMSE computation (for sklearn versions that lack squared=False)
    mse_log = mean_squared_error(yr_val, pr)
    rmse_log = mse_log ** 0.5
    print("Regression RMSE (log-price, val):", rmse_log)

    # Back-transform to price-space and measure RMSE there as well
    price_val = np.expm1(yr_val)
    price_hat = np.expm1(pr)
    price_hat = np.clip(price_hat, 0, None)
    mse_price = mean_squared_error(price_val, price_hat)
    rmse_price = mse_price ** 0.5
    print("Regression RMSE (price, val):", rmse_price)

    reg_path = os.path.join(OUT_DIR, "lgbm_price_regressor.pkl")
    joblib.dump(reg, reg_path)
    print("Saved regressor ->", reg_path)

print("Training script finished.")
