#!/usr/bin/env python3
"""
inference_auction_v4.py

Uses enhanced_players_features.csv + cat_maps.json + num_stats.json + models_v3 (or v3 names)
Maps unseen categorical values to the most common training category (safer).
Outputs predictions_v4.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np
import joblib
import sys

PROC = Path(r"D:\RR_Hackathon_2025\data\processed")
MODELS = Path(r"D:\RR_Hackathon_2025\models")
TRAIN = PROC / "model_data.csv"
ENHANCED = PROC / "enhanced_players_features.csv"
OUT = PROC / "predictions_v4.csv"
CAT_MAP = MODELS / "cat_maps.json"
NUM_STATS = MODELS / "num_stats.json"
CLF = MODELS / "lgbm_sold_classifier_v3.pkl"
REG = MODELS / "lgbm_price_regressor_v3.pkl"

def load_json(p):
    with open(p,'r',encoding='utf8') as f:
        return json.load(f)

def most_common(mapping_list):
    # fallback is index 0 (first category in training order). We will return index 0 if unseen.
    return 0 if len(mapping_list)>0 else -1

def apply_safe_map(series, mapping_list):
    mp = {cat:i for i,cat in enumerate(mapping_list)}
    default_idx = most_common(mapping_list)
    def _map(val):
        v = "NA" if (pd.isna(val) or str(val)=="") else str(val)
        return mp[v] if v in mp else default_idx
    return series.fillna("NA").astype(str).map(_map).astype(int)

def main(args):
    # sanity
    for p in [TRAIN, ENHANCED, CAT_MAP, NUM_STATS, CLF, REG]:
        if not p.exists():
            print("[ERROR] Missing file:", p)
            return

    train = pd.read_csv(TRAIN, low_memory=False)
    cat_maps = load_json(CAT_MAP)
    num_stats = load_json(NUM_STATS)

    # reconstruct feature_cols used in training (exclude targets)
    exclude = {'player_id','player_name','sold_price','sold','price','log_price','log_price_lakhs','price_lakhs',
               'player_name_players','player_name_bow'}
    feature_cols = [c for c in train.columns if c not in exclude]

    players = pd.read_csv(ENHANCED, low_memory=False)
    if 'year' not in players.columns:
        players['year'] = 2025

    # Apply safe categorical mapping for each categorical in cat_maps if used
    for c, mapping_list in cat_maps.items():
        if c in feature_cols:
            if c not in players.columns:
                players[c] = "NA"
            players[c] = apply_safe_map(players[c], mapping_list)

    # Coerce numeric features
    for c in feature_cols:
        if c not in players.columns:
            players[c] = 0
        else:
            players[c] = pd.to_numeric(players[c], errors='coerce').fillna(0)

    # ensure same order
    X = players[feature_cols].fillna(0)

    # load models
    clf = joblib.load(CLF)
    reg = joblib.load(REG)

    # predict
    try:
        p_sold = clf.predict_proba(X)[:,1]
    except Exception:
        p_sold = clf.predict(X).astype(float)

    log_price_lakhs = reg.predict(X)
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
    print("[DONE] Saved ->", OUT)
    print(out.head(30).to_string(index=False))

if __name__ == "__main__":
    main(None)
