#!/usr/bin/env python3
"""
inference_auction_v2.py
Robust inference script:
 - ensures all features exist (adds year=2025 default)
 - reconstructs factorize mappings from model_data.csv
 - maps unseen categories to a new index (len(map)) instead of -1
 - coerces numerics and reports diagnostics
 - saves predictions_v2.csv
"""
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PROC = Path(r"D:\RR_Hackathon_2025\data\processed")
MODELS = Path(r"D:\RR_Hackathon_2025\models")
TRAIN_MODEL_DATA = PROC / "model_data.csv"
PLAYERS = PROC / "final_players_features.csv"
OUT = PROC / "predictions_v2.csv"

# Load training model data and players features
train = pd.read_csv(TRAIN_MODEL_DATA, low_memory=False)
players = pd.read_csv(PLAYERS, low_memory=False)

# Recreate feature list used during training
exclude = {'player_id','player_name','sold_price','sold','price','log_price'}
feature_cols = [c for c in train.columns if c not in exclude]

# Ensure 'year' exists in players (training had year)
if 'year' not in players.columns:
    players['year'] = 2025
    print("[INFO] Added default 'year'=2025 to players features.")

# Build encoding maps for object columns from training data
encoding_maps = {}
for c in feature_cols:
    if train[c].dtype == object:
        cats = pd.Series(train[c].fillna("NA").astype(str).unique())
        mapping = {cat: i for i, cat in enumerate(cats)}  # factorize order
        encoding_maps[c] = mapping

# Apply encodings to players; unseen -> new index (=len(mapping))
for c, mapping in encoding_maps.items():
    if c not in players.columns:
        players[c] = "NA"
    # convert to str
    col = players[c].fillna("NA").astype(str)
    # map unseen values to new index
    def map_with_new(val, mp=mapping):
        if val in mp:
            return mp[val]
        else:
            return len(mp)  # new unseen category
    players[c] = col.map(map_with_new).astype(int)

# Ensure numeric columns are numeric
for c in feature_cols:
    if c in players.columns:
        if train[c].dtype != object:
            players[c] = pd.to_numeric(players[c], errors='coerce').fillna(0)

# Ensure all feature cols exist
for c in feature_cols:
    if c not in players.columns:
        players[c] = 0

# Diagnostics: feature variance & constants
variances = {}
constant_cols = []
for c in feature_cols:
    try:
        var = float(pd.to_numeric(players[c], errors='coerce').var())
    except Exception:
        var = 0.0
    variances[c] = var
    if pd.to_numeric(players[c], errors='coerce').nunique(dropna=False) <= 1:
        constant_cols.append(c)

# Print diagnostics summary
print("[DIAG] features count:", len(feature_cols))
print("[DIAG] constant features in players (single unique):", constant_cols)
# show top 10 most variable features
sorted_vars = sorted(variances.items(), key=lambda x: x[1], reverse=True)
print("[DIAG] top 10 feature variances:")
for k,v in sorted_vars[:10]:
    print(f"  {k}: var={v:.6f}")

# Load models
clf = joblib.load(MODELS / "lgbm_sold_classifier.pkl")
reg = joblib.load(MODELS / "lgbm_price_regressor.pkl")

# Build X in the same column order
X = players[feature_cols].fillna(0)

# Predict
try:
    p_sold = clf.predict_proba(X)[:,1]
except Exception:
    p_sold = clf.predict(X).astype(float)

log_price_pred = reg.predict(X)
price_if_sold = np.expm1(log_price_pred)
price_if_sold = np.clip(price_if_sold, 0, None)
expected_price = p_sold * price_if_sold

# Output
out = players[['player_id','player_name']].copy() if 'player_id' in players.columns or 'player_name' in players.columns else pd.DataFrame(index=players.index)
if 'player_id' in players.columns:
    out['player_id'] = players['player_id']
if 'player_name' in players.columns:
    out['player_name'] = players['player_name']

out['p_sold'] = p_sold
out['pred_price_if_sold'] = price_if_sold
out['expected_price'] = expected_price

out = out.sort_values('expected_price', ascending=False).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"[DONE] Saved predictions -> {OUT}")
print(out.head(30).to_string(index=False))
