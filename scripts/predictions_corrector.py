#!/usr/bin/env python3
"""
predictions_corrector.py

Robust corrected predictions builder.

- Detects player list/base price columns in Player List.xlsx (various names).
- Detects predicted price column in predictions CSV and normalizes units.
- Produces:
    - corrected_predictions.csv with both model_pred_only (rupees) and corrected_pred_price_if_sold (rupees)
    - expected_price (rupees) and expected_price_cr (crores)
    - short correction_report.txt and debug name lists if merges fail.

Usage (examples):
  # default behavior (assuming project layout described in README)
  python predictions_corrector.py

  # explicit paths (recommended)
  python predictions_corrector.py \
    --playerlist "D:/RR_Hackathon_2025/data/RAW/Player List.xlsx" \
    --predictions "D:/RR_Hackathon_2025/data/processed/predictions_catboost_v5.csv" \
    --output "D:/RR_Hackathon_2025/data/processed/corrected_predictions.csv"
"""
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

# ---------- helpers ----------
def find_col_ci(df, candidates):
    """Case-insensitive column finder. Returns actual column name or None."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in lower_map:
            return lower_map[k]
    return None

def norm_name(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip().lower()
    # drop punctuation but keep unicode letters/numbers/space
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_price_column(pred_df):
    """Pick best candidate column for predicted price. Return (col_name, unit) where unit is 'rupees' or 'crores'."""
    # common names to try
    try_names = [
        "pred_price_if_sold","pred_price","pred_price_raw","pred_price_cr","pred_price_crore",
        "predicted_price","predicted_price_if_sold","price_pred","price_if_sold","price"
    ]
    for name in try_names:
        col = find_col_ci(pred_df, [name])
        if col is not None:
            # try numeric
            vals = pd.to_numeric(pred_df[col], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            vmax = vals.max(skipna=True)
            if pd.isna(vmax):
                continue
            # heuristic: if max > 1e6 => values likely in rupees; else crores
            if vmax > 1e6:
                return col, "rupees"
            else:
                return col, "crores"
    # fallback: choose numeric column with largest max
    numeric_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        best = max(numeric_cols, key=lambda c: pd.to_numeric(pred_df[c], errors="coerce").max() if pd.to_numeric(pred_df[c], errors="coerce").notna().any() else -1)
        vmax = pd.to_numeric(pred_df[best], errors="coerce").max()
        if pd.isna(vmax):
            return None, None
        return (best, "rupees") if vmax > 1e6 else (best, "crores")
    return None, None

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--playerlist", "-p", type=str, default=None,
                   help="Path to Player List Excel (default: D:/RR_Hackathon_2025/data/RAW/Player List.xlsx)")
    p.add_argument("--predictions", "-r", type=str, default=None,
                   help="Path to predictions CSV (default: D:/RR_Hackathon_2025/data/processed/predictions_catboost_v5.csv -> fallback to RAW/)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output corrected CSV path (default: D:/RR_Hackathon_2025/data/processed/corrected_predictions.csv)")
    args = p.parse_args()

    # sensible defaults (absolute paths)
    PROJECT_DIR = Path("D:/RR_Hackathon_2025")
    default_playerlist = PROJECT_DIR / "data" / "RAW" / "Player List.xlsx"
    default_predictions_processed = PROJECT_DIR / "data" / "processed" / "predictions_catboost_v5.csv"
    default_predictions_raw = PROJECT_DIR / "data" / "RAW" / "predictions_catboost_v5.csv"
    default_output = PROJECT_DIR / "data" / "processed" / "corrected_predictions.csv"

    PLAYER_LIST_PATH = Path(args.playerlist) if args.playerlist else default_playerlist
    # prefer processed; fallback to raw
    if args.predictions:
        PREDICTIONS_PATH = Path(args.predictions)
    else:
        PREDICTIONS_PATH = default_predictions_processed if default_predictions_processed.exists() else default_predictions_raw

    OUTPUT_PATH = Path(args.output) if args.output else default_output

    print("[INFO] Player List path:", PLAYER_LIST_PATH)
    print("[INFO] Predictions path:", PREDICTIONS_PATH)
    print("[INFO] Output path:", OUTPUT_PATH)

    if not PLAYER_LIST_PATH.exists():
        raise FileNotFoundError(f"Player list not found at {PLAYER_LIST_PATH}")
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions CSV not found at {PREDICTIONS_PATH}")

    # read files
    try:
        plist = pd.read_excel(PLAYER_LIST_PATH, engine="openpyxl")
    except Exception as e:
        print("[ERROR] Could not read Player List:", e)
        raise

    pred = pd.read_csv(PREDICTIONS_PATH, low_memory=False)

    print(f"[INFO] Player list columns: {plist.columns.tolist()}")
    print(f"[INFO] Predictions columns: {pred.columns.tolist()}")

    # detect player/base/category columns in Player List
    player_col = find_col_ci(plist, ["player_name", "player name", "name", "player"])
    base_col   = find_col_ci(plist, ["base_price", "base price", "baseprice", "base", "base_price_cr"])
    cat_col    = find_col_ci(plist, ["category", "player_category", "type", "nat", "nationality", "role"])

    if player_col is None:
        raise KeyError("Could not find player name column in Player List. Expected candidate columns like 'Player Name'.")
    if base_col is None:
        raise KeyError("Could not find base price column in Player List. Expected 'Base Price' or similar.")

    # rename to standard names and normalize
    plist = plist.rename(columns={player_col: "player_name", base_col: "base_price"})
    if cat_col:
        plist = plist.rename(columns={cat_col: "category"})
    else:
        plist["category"] = np.nan

    plist["player_name"] = plist["player_name"].astype(str).str.strip()
    plist["player_name_norm"] = plist["player_name"].apply(norm_name)
    # drop duplicate normalized names (keep first)
    plist = plist.drop_duplicates(subset=["player_name_norm"], keep="first").reset_index(drop=True)

    # predictions: detect name column
    pred_name_col = find_col_ci(pred, ["player_name", "player name", "name", "player"])
    if pred_name_col is None:
        raise KeyError("Could not find player name column in predictions CSV.")
    pred = pred.rename(columns={pred_name_col: "player_name"})
    pred["player_name"] = pred["player_name"].astype(str).str.strip()
    pred["player_name_norm"] = pred["player_name"].apply(norm_name)

    print(f"[INFO] Player list rows: {len(plist)}; predictions rows: {len(pred)}")

    # merge inner: keep only players present in playerlist (RR eligible)
    merged = pred.merge(plist[["player_name_norm", "player_name", "base_price", "category"]],
                        on="player_name_norm", how="inner", suffixes=("_pred", "_plist"))

    print(f"[INFO] Matched rows after merge: {len(merged)}")
    if merged.empty:
        print("[WARN] No matches between player list and predictions. Saving debug name lists to processed folder.")
        outdir = OUTPUT_PATH.parent
        outdir.mkdir(parents=True, exist_ok=True)
        pred[['player_name', 'player_name_norm']].drop_duplicates().to_csv(outdir / "pred_names_debug.csv", index=False)
        plist[['player_name', 'player_name_norm']].drop_duplicates().to_csv(outdir / "playerlist_names_debug.csv", index=False)
        return

    # detect price column and units
    price_col, price_unit = detect_price_column(pred)
    if price_col is None:
        print("[WARN] Could not detect a price-like column in predictions. Setting model_pred_only = 0")
        merged["model_pred_only"] = 0.0
        detected_price_col = None
        detected_price_unit = None
    else:
        detected_price_col = price_col
        detected_price_unit = price_unit
        print(f"[INFO] Detected predictions price column: '{detected_price_col}' (assumed unit: {detected_price_unit})")
        # move/rename that col from merged if present (col might have original name)
        # merged already contains columns from pred; ensure we use the merged column name
        # if the detected col is in merged, use it; otherwise fallback to numeric detection later
        if detected_price_col in merged.columns:
            merged["pred_price_raw"] = pd.to_numeric(merged[detected_price_col], errors="coerce").fillna(0.0)
        else:
            # fallback: try to pick any numeric in merged that wasn't from plist
            numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ["base_price"]]
            if numeric_cols:
                # pick the largest numeric column by max
                best = max(numeric_cols, key=lambda c: pd.to_numeric(merged[c], errors="coerce").max() if pd.to_numeric(merged[c], errors="coerce").notna().any() else -1)
                merged["pred_price_raw"] = pd.to_numeric(merged[best], errors="coerce").fillna(0.0)
                detected_price_col = best
            else:
                merged["pred_price_raw"] = 0.0

    # normalize pred units into RUPEES (model_pred_only column)
    if "pred_price_raw" in merged.columns:
        if detected_price_unit == "crores":
            merged["model_pred_only"] = merged["pred_price_raw"] * 1e7
        elif detected_price_unit == "rupees":
            merged["model_pred_only"] = merged["pred_price_raw"]
        else:
            # decide heuristically if values are small -> crores else rupees
            v = merged["pred_price_raw"].max()
            if pd.notna(v) and v <= 100:
                merged["model_pred_only"] = merged["pred_price_raw"] * 1e7
            else:
                merged["model_pred_only"] = merged["pred_price_raw"]
    else:
        merged["model_pred_only"] = 0.0

    # p_sold probability (if exists)
    merged["p_sold"] = pd.to_numeric(merged.get("p_sold", 0.0), errors="coerce").fillna(0.0)

    # base_price normalization: assume Player List base_price is in crores (1.0 => 1 crore),
    # but if values are huge (>1000) treat them as rupees already.
    merged["base_price"] = pd.to_numeric(merged.get("base_price", 0.0), errors="coerce").fillna(0.0)
    if merged["base_price"].max() > 1000:
        merged["base_price_rs"] = merged["base_price"]
        base_unit = "rupees"
    else:
        merged["base_price_rs"] = merged["base_price"] * 1e7
        base_unit = "crores"

    # corrected predicted price (apply floor)
    merged["corrected_pred_price_if_sold"] = merged[["model_pred_only", "base_price_rs"]].max(axis=1)

    # expected price (rupees) and crores column for readability
    merged["expected_price"] = merged["corrected_pred_price_if_sold"] * merged["p_sold"]
    merged["expected_price_cr"] = merged["expected_price"] / 1e7  # crores

    # save processed output
    outdir = OUTPUT_PATH.parent
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"[DONE] Saved corrected predictions -> {OUTPUT_PATH}")

    # diagnostics: top-20 by expected_price
    top = merged.sort_values("expected_price", ascending=False).head(20)
    # choose sensible columns to display if they exist
    cols_pref = [
        "player_id", "player_name_plist", "player_name_pred", "player_name",
        "category", "base_price", "base_price_rs",
        "model_pred_only", "corrected_pred_price_if_sold", "p_sold",
        "expected_price", "expected_price_cr"
    ]
    available = [c for c in cols_pref if c in merged.columns]
    print("\nTop 20 by expected_price (after correction):")
    if len(available) > 0:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
            print(top[available].to_string(index=False))
    else:
        print(top.head(10).to_string(index=False))

    # Save short report
    report_path = outdir / "correction_report.txt"
    with open(report_path, "w", encoding="utf8") as f:
        f.write(f"Players in player list: {len(plist)}\n")
        f.write(f"Predictions rows: {len(pred)}\n")
        f.write(f"Matched rows: {len(merged)}\n")
        f.write("Detected / used columns:\n")
        f.write(f" - predictions price column (detected) -> {detected_price_col}\n")
        f.write(f" - predictions price unit (detected) -> {detected_price_unit}\n")
        f.write(f" - player_list player column -> {player_col}\n")
        f.write(f" - player_list base price column -> {base_col}\n")
        f.write(f" - player_list category column -> {cat_col}\n")
        f.write("\nSaved corrected_predictions.csv at: " + str(OUTPUT_PATH) + "\n")

    print(f"[INFO] Short report saved -> {report_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR] Exception during run:", e)
        sys.exit(1)
