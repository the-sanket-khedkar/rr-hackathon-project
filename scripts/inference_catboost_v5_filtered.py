#!/usr/bin/env python3
"""
inference_catboost_v5_filtered.py

Runs catboost inference using a filtered players CSV (auction-only).
Default players file: data/processed/enhanced_players_features_auction.csv
Writes: data/processed/predictions_catboost_v5_auction.csv

Usage:
  python inference_catboost_v5_filtered.py
  python inference_catboost_v5_filtered.py --players PATH/TO/filtered.csv --out PATH/TO/out.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import sys

ROOT = Path(r"D:\RR_Hackathon_2025")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
META = MODELS / "catboost_meta_v5.json"
CLF_MODEL = MODELS / "catboost_classifier_v5.cbm"
REG_MODEL = MODELS / "catboost_regressor_v5.cbm"
DEFAULT_PLAYERS = PROC / "enhanced_players_features_auction.csv"
DEFAULT_OUT = PROC / "predictions_catboost_v5_auction.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', help='Filtered players CSV (auction-only)', default=str(DEFAULT_PLAYERS))
    parser.add_argument('--out', help='Output CSV', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    players_path = Path(args.players)
    out_path = Path(args.out)

    for p in [META, CLF_MODEL, REG_MODEL, players_path]:
        if not p.exists():
            print("[ERROR] Missing required file:", p)
            return

    meta = json.loads(META.read_text(encoding='utf8'))
    feature_cols = meta['feature_cols']
    cat_cols = meta['cat_cols']

    players = pd.read_csv(players_path, low_memory=False)
    if 'year' not in players.columns:
        players['year'] = 2025

    # Ensure columns exist
    for c in feature_cols:
        if c not in players.columns:
            players[c] = np.nan

    X = players[feature_cols].copy()
    for c in feature_cols:
        if c in cat_cols:
            X[c] = X[c].fillna("NA").astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

    cat_feature_indices = [i for i, c in enumerate(feature_cols) if c in cat_cols]

    clf = CatBoostClassifier()
    clf.load_model(str(CLF_MODEL))
    reg = CatBoostRegressor()
    reg.load_model(str(REG_MODEL))

    pool = Pool(X, cat_features=cat_feature_indices if len(cat_feature_indices)>0 else None)

    p_sold = clf.predict_proba(pool)[:,1]
    log_price_lakhs = reg.predict(pool)
    price_if_sold = np.expm1(log_price_lakhs) * 1e5
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
    out.to_csv(out_path, index=False)
    print("[DONE] Saved auction-only predictions ->", out_path)
    print(out.head(40).to_string(index=False))

if __name__ == '__main__':
    main()
