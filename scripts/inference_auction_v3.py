#!/usr/bin/env python3
"""
inference_auction_v3.py

Usage:
    python inference_auction_v3.py
    python inference_auction_v3.py --players PATH_TO_PLAYERS_CSV --out PATH_TO_OUTPUT

Assumptions:
 - train_baseline_v3.py already ran and produced:
    D:\RR_Hackathon_2025\models\cat_maps.json
    D:\RR_Hackathon_2025\models\num_stats.json
    D:\RR_Hackathon_2025\models\lgbm_sold_classifier_v3.pkl
    D:\RR_Hackathon_2025\models\lgbm_price_regressor_v3.pkl
 - D:\RR_Hackathon_2025\data\processed\model_data.csv exists (used to compute feature_cols)
 - Players features file defaults to D:\RR_Hackathon_2025\data\processed\final_players_features.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys

# ---------- config ----------
PROC = Path(r"D:\RR_Hackathon_2025\data\processed")
MODELS = Path(r"D:\RR_Hackathon_2025\models")
TRAIN_MODEL_DATA = PROC / "model_data.csv"
DEFAULT_PLAYERS = PROC / "final_players_features.csv"
OUT = PROC / "predictions_v3.csv"
CAT_MAPS = MODELS / "cat_maps.json"
NUM_STATS = MODELS / "num_stats.json"
CLF_MODEL = MODELS / "lgbm_sold_classifier_v3.pkl"
REG_MODEL = MODELS / "lgbm_price_regressor_v3.pkl"

# ---------- helpers ----------
def load_json(p):
    with open(p, 'r', encoding='utf8') as f:
        return json.load(f)

def norm_name(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def apply_cat_map(series, mapping_list):
    """
    mapping_list: list of categories in training order
    returns integer Series where category -> index in mapping_list,
    unseen -> len(mapping_list)
    """
    mp = {cat: i for i, cat in enumerate(mapping_list)}
    def _map_val(v):
        v = "NA" if (pd.isna(v) or str(v)=="" ) else str(v)
        return mp[v] if v in mp else len(mapping_list)
    return series.fillna("NA").astype(str).map(_map_val).astype(int)

# ---------- main ----------
def main(args):
    players_path = Path(args.players) if args.players else DEFAULT_PLAYERS
    out_path = Path(args.out) if args.out else OUT

    # sanity checks
    for p in [TRAIN_MODEL_DATA, CAT_MAPS, NUM_STATS, CLF_MODEL, REG_MODEL]:
        if not p.exists():
            print(f"[ERROR] Required file not found: {p}", file=sys.stderr)
            return

    # load training model data to recover feature columns
    train = pd.read_csv(TRAIN_MODEL_DATA, low_memory=False)
    exclude = {'player_id','player_name','sold_price','sold','price','log_price','log_price_lakhs','price_lakhs'}
    # also exclude name-derived columns used earlier
    name_cols = ['player_name_players','player_name_bow']
    exclude.update(name_cols)
    feature_cols = [c for c in train.columns if c not in exclude]
    print(f"[INFO] Reconstructed feature_cols (count={len(feature_cols)})")

    # load mappings & numeric stats produced by train_baseline_v3
    cat_maps = load_json(CAT_MAPS)
    num_stats = load_json(NUM_STATS)

    # load players features
    if not players_path.exists():
        print(f"[ERROR] players features file not found: {players_path}", file=sys.stderr)
        return
    players = pd.read_csv(players_path, low_memory=False)
    print(f"[INFO] Loaded players features: {players.shape}")

    # add year if missing
    if 'year' not in players.columns:
        players['year'] = 2025
        print("[INFO] Added default year=2025 to players")

    # normalize player_name if present (keeps consistency with earlier merges)
    if 'player_name' in players.columns:
        players['player_name_norm'] = players['player_name'].apply(norm_name)
    else:
        players['player_name_norm'] = ""

    # Apply categorical mappings (only for columns present in cat_maps)
    for c, mapping_list in cat_maps.items():
        if c not in feature_cols:
            continue
        if c not in players.columns:
            # create column with NA
            players[c] = "NA"
        # apply mapping; unseen -> new index len(mapping_list)
        players[c] = apply_cat_map(players[c], mapping_list)

    # Coerce numeric columns
    for c in feature_cols:
        if c in cat_maps:
            # already encoded
            continue
        if c not in players.columns:
            players[c] = 0
        else:
            # coerce to numeric
            players[c] = pd.to_numeric(players[c], errors='coerce').fillna(0)

    # Ensure all feature_cols exist
    for c in feature_cols:
        if c not in players.columns:
            players[c] = 0

    # Build X in same order
    X = players[feature_cols].fillna(0)

    # diagnostics
    const_cols = [c for c in feature_cols if X[c].nunique(dropna=False) <= 1]
    print("[DIAG] constant features count:", len(const_cols))
    if len(const_cols) > 0:
        print("[DIAG] constant features:", const_cols[:20])

    # load models
    clf = joblib.load(CLF_MODEL)
    reg = joblib.load(REG_MODEL)
    print("[INFO] Loaded models.")

    # predict
    try:
        p_sold = clf.predict_proba(X)[:,1]
    except Exception:
        p_sold = clf.predict(X).astype(float)

    # regression: reg predicts log_price_lakhs (training target)
    log_price_lakhs = reg.predict(X)
    # back-transform price in rupees: price = expm1(log_price_lakhs) * 1e5
    price_if_sold = np.expm1(log_price_lakhs) * 1e5
    price_if_sold = np.clip(price_if_sold, 0, None)
    expected_price = p_sold * price_if_sold

    # build output
    out = pd.DataFrame()
    if 'player_id' in players.columns:
        out['player_id'] = players['player_id']
    if 'player_name' in players.columns:
        out['player_name'] = players['player_name']

    out['p_sold'] = p_sold
    out['pred_price_if_sold'] = price_if_sold
    out['expected_price'] = expected_price

    # save
    out = out.sort_values('expected_price', ascending=False).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] Saved predictions -> {out_path}")
    print(out.head(30).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', help="players features CSV (default final_players_features.csv)")
    parser.add_argument('--out', help="output CSV (default predictions_v3.csv)")
    args = parser.parse_args()
    main(args)
