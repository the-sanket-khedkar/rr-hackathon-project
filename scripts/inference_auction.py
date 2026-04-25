#!/usr/bin/env python3
"""
inference_auction.py

Usage (from project root or scripts folder):
    python inference_auction.py
    python inference_auction.py --players D:\RR_Hackathon_2025\data\processed\final_players_features.csv
    python inference_auction.py --out D:\RR_Hackathon_2025\data\processed\predictions_myrun.csv

What it does:
 - Loads models from D:\RR_Hackathon_2025\models:
      lgbm_sold_classifier.pkl
      lgbm_price_regressor.pkl
 - Loads training model_data.csv to reconstruct categorical encodings & feature list
 - Loads the players file (default final_players_features.csv)
 - Aligns features and makes predictions
 - Writes predictions CSV and prints top 50 by expected price
"""
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# ---------- CONFIG ----------
DEFAULT_PROC = Path(r"D:\RR_Hackathon_2025\data\processed")
DEFAULT_MODELS = Path(r"D:\RR_Hackathon_2025\models")
TRAIN_MODEL_DATA = DEFAULT_PROC.joinpath("model_data.csv")
DEFAULT_PLAYERS = DEFAULT_PROC.joinpath("final_players_features.csv")
OUT_PRED = DEFAULT_PROC.joinpath("predictions.csv")

# ---------- Helpers ----------
def build_encoding_maps(train_df, feature_cols):
    """
    For each categorical column in feature_cols (dtype object in train_df),
    build a mapping dict {category: int} consistent with how training factorize() worked.
    Unknown categories will map to -1 at inference time.
    Returns: dict col -> Series mapping
    """
    maps = {}
    for c in feature_cols:
        if train_df[c].dtype == object:
            cats = pd.Series(train_df[c].fillna("NA").astype(str).unique())
            # mimic pd.factorize: categories get indexes in appearance order
            mapping = {cat: i for i, cat in enumerate(cats)}
            maps[c] = mapping
    return maps

def apply_encodings(df, maps):
    """Apply mapping dicts to df in-place, unknown -> -1"""
    for c, mp in maps.items():
        if c not in df.columns:
            # create missing column with default 'NA'
            df[c] = "NA"
        # convert to string and map
        df[c] = df[c].fillna("NA").astype(str).map(mp).fillna(-1).astype(int)

def ensure_feature_columns(df, feature_cols):
    """Ensure df has all columns in feature_cols; add missing with zeros."""
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df

# ---------- Main ----------
def main(args):
    # Paths
    players_path = Path(args.players) if args.players else DEFAULT_PLAYERS
    model_data_path = Path(args.model_data) if args.model_data else TRAIN_MODEL_DATA
    models_dir = Path(args.models_dir) if args.models_dir else DEFAULT_MODELS
    out_path = Path(args.out) if args.out else OUT_PRED

    # Load trained models
    clf_path = models_dir.joinpath("lgbm_sold_classifier.pkl")
    reg_path = models_dir.joinpath("lgbm_price_regressor.pkl")
    if not clf_path.exists() or not reg_path.exists():
        raise FileNotFoundError(f"Model files not found in {models_dir}. Expected {clf_path.name} and {reg_path.name}")

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)

    # Load training model_data to reconstruct encodings & feature set
    if not model_data_path.exists():
        raise FileNotFoundError(f"Training model_data not found at {model_data_path}.")
    train_df = pd.read_csv(model_data_path, low_memory=False)

    # Reconstruct feature list used in training
    exclude = {'player_id','player_name','sold_price','sold','price','log_price'}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    # Build encoding maps from train_df for object columns
    encoding_maps = build_encoding_maps(train_df, feature_cols)

    # Load players file
    if not players_path.exists():
        raise FileNotFoundError(f"Players/features file not found at {players_path}")
    players_df = pd.read_csv(players_path, low_memory=False)

    # If players file contains 'player_id' or 'player_name', keep them for output
    id_cols = []
    if 'player_id' in players_df.columns:
        id_cols.append('player_id')
    if 'player_name' in players_df.columns:
        id_cols.append('player_name')

    # Ensure features exist in players_df
    players_df = ensure_feature_columns(players_df, feature_cols)

    # For any columns that are numeric in train_df but object in players (or vice-versa), coerce types
    # Apply encodings
    apply_encodings(players_df, encoding_maps)

    # Build X in same column order
    X = players_df[feature_cols].fillna(0)

    # Predict probabilities and log-price
    try:
        p_sold = clf.predict_proba(X)[:,1]
    except Exception as e:
        # If classifier doesn't support predict_proba (unlikely), fallback to predict
        p_sold = clf.predict(X).astype(float)

    # For regression, predict log-price then back-transform
    log_price_pred = reg.predict(X)
    # In case reg predicts negative numbers or something weird, clip to reasonable range
    # back-transform to price-space
    price_pred_if_sold = np.expm1(log_price_pred)
    price_pred_if_sold = np.clip(price_pred_if_sold, 0, None)

    expected_price = p_sold * price_pred_if_sold

    # Build output DataFrame
    out = players_df[id_cols].copy() if id_cols else pd.DataFrame(index=players_df.index)
    out['p_sold'] = p_sold
    out['pred_price_if_sold'] = price_pred_if_sold
    out['expected_price'] = expected_price

    # Keep some useful player columns if present
    keep = ['player_name','player_id']
    for k in keep:
        if k in players_df.columns and k not in out.columns:
            out[k] = players_df[k]

    # Sort by expected_price descending
    out = out.sort_values('expected_price', ascending=False).reset_index(drop=True)

    # Save results
    out.to_csv(out_path, index=False)
    print(f"Saved predictions -> {out_path}")
    print("\nTop 30 by expected_price:")
    pd.set_option('display.max_rows', 30)
    print(out.head(30).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auction inference script")
    parser.add_argument('--players', help="CSV of player features to predict (default: processed/final_players_features.csv)")
    parser.add_argument('--model_data', help="Training model_data.csv used to reconstruct encodings (default: processed/model_data.csv)")
    parser.add_argument('--models_dir', help="Directory with trained models", default=str(DEFAULT_MODELS))
    parser.add_argument('--out', help="Output CSV path", default=str(OUT_PRED))
    args = parser.parse_args()
    main(args)
