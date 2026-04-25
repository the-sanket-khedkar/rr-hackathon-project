#!/usr/bin/env python3
"""
inference_catboost_v5.py
- Loads catboost models (classifier + regressor) saved by train_catboost_v5.py
- Loads enhanced players features and produces predictions_v5.csv
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ROOT = Path(r"D:\RR_Hackathon_2025")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
META = MODELS / "catboost_meta_v5.json"
CLF_MODEL = MODELS / "catboost_classifier_v5.cbm"
REG_MODEL = MODELS / "catboost_regressor_v5.cbm"
ENHANCED = PROC / "enhanced_players_features.csv"
OUT = PROC / "predictions_catboost_v5.csv"

# sanity
for p in [META, CLF_MODEL, REG_MODEL, ENHANCED]:
    if not p.exists():
        print("[ERROR] Missing file:", p)
        raise SystemExit(1)

meta = json.loads(META.read_text(encoding='utf8'))
feature_cols = meta['feature_cols']
cat_cols = meta['cat_cols']

# load enhanced players
players = pd.read_csv(ENHANCED, low_memory=False)

# ensure year exists
if 'year' not in players.columns:
    players['year'] = 2025

# ensure all feature_cols exist in players; if missing, add with 0/NA
for c in feature_cols:
    if c not in players.columns:
        players[c] = np.nan

# prepare X with same types: categorical columns kept as strings
X = players[feature_cols].copy()
for c in feature_cols:
    if c in cat_cols:
        X[c] = X[c].fillna("NA").astype(str)
    else:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

# create cat feature indices (catboost expects column indices)
cat_feature_indices = [i for i, c in enumerate(feature_cols) if c in cat_cols]

# load models
clf = CatBoostClassifier()
clf.load_model(str(CLF_MODEL))
reg = CatBoostRegressor()
reg.load_model(str(REG_MODEL))

# create pools
pool = Pool(X, cat_features=cat_feature_indices if len(cat_feature_indices)>0 else None)

# predict
p_sold = clf.predict_proba(pool)[:,1]
log_price_lakhs = reg.predict(pool)
price_if_sold = np.expm1(log_price_lakhs) * 1e5
price_if_sold = np.clip(price_if_sold, 0, None)
expected = p_sold * price_if_sold

out = pd.DataFrame()
if 'player_id' in players.columns:
    out['player_id'] = players['player_id']
if 'player_name' in players.columns:
    out['player_name'] = players['player_name']
out['p_sold'] = p_sold
out['pred_price_if_sold'] = price_if_sold
out['expected_price'] = expected

out = out.sort_values('expected_price', ascending=False).reset_index(drop=True)
out.to_csv(OUT, index=False)
print("[DONE] Saved predictions ->", OUT)
print(out.head(30).to_string(index=False))
