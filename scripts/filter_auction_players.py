#!/usr/bin/env python3
"""
filter_auction_players.py

Filters enhanced_players_features.csv to only players present in Player List.xlsx (auction register).
Output -> D:\RR_Hackathon_2025\data\processed\enhanced_players_features_auction.csv

Matching strategy:
 1) Exact normalized match on player_name (lower, strip)
 2) Fallback: fuzzy matching using difflib.get_close_matches (stdlib) with cutoff threshold
    - Only used if exact match fails.
    - Prints fuzzy suggestions for manual inspection.

Usage:
  python filter_auction_players.py
"""
from pathlib import Path
import pandas as pd
import difflib
import argparse
import sys

ROOT = Path(r"D:\RR_Hackathon_2025")
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

ENHANCED = PROC / "enhanced_players_features.csv"
PLAYER_LIST_XLSX = RAW / "Player List.xlsx"
OUT = PROC / "enhanced_players_features_auction.csv"

# parameters
FUZZY_CUTOFF = 0.85   # threshold for difflib SequenceMatcher ratio (0..1)
MAX_SUGGESTIONS = 3

def norm_name(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def find_fuzzy(name, choices, cutoff=FUZZY_CUTOFF, max_suggestions=MAX_SUGGESTIONS):
    # difflib.get_close_matches returns suggestions based on seqmatch
    # but it uses a cutoff on similarity; we'll return list of tuples (choice, ratio)
    matches = difflib.get_close_matches(name, choices, n=max_suggestions, cutoff=cutoff)
    return matches

def main():
    if not ENHANCED.exists():
        print("[ERROR] Enhanced players file not found:", ENHANCED, file=sys.stderr)
        return
    if not PLAYER_LIST_XLSX.exists():
        print("[ERROR] Player List.xlsx not found (expected at):", PLAYER_LIST_XLSX, file=sys.stderr)
        return

    enh = pd.read_csv(ENHANCED, low_memory=False)
    try:
        auction = pd.read_excel(PLAYER_LIST_XLSX, engine='openpyxl')
    except Exception:
        # fallback if openpyxl isn't available
        auction = pd.read_excel(PLAYER_LIST_XLSX)

    # Expect auction list to have a player_name column - try to find it
    cand_cols = [c for c in auction.columns if 'name' in c.lower()]
    if not cand_cols:
        print("[WARN] Could not find a 'player_name' column in Player List.xlsx; printing columns:", auction.columns.tolist())
        # attempt to use first column
        auction['player_name'] = auction.iloc[:,0].astype(str)
    else:
        auction['player_name'] = auction[cand_cols[0]].astype(str)

    # Normalize names
    auction['player_name_norm'] = auction['player_name'].apply(norm_name)
    enh['player_name_norm'] = enh['player_name'].apply(norm_name)

    auction_names = auction['player_name_norm'].dropna().unique().tolist()
    enh_names = enh['player_name_norm'].dropna().unique().tolist()

    # Exact matches
    matched = enh[enh['player_name_norm'].isin(auction_names)].copy()
    exact_matched_count = matched.shape[0]

    # Names in auction that were not matched exactly (for diagnostics)
    auction_only = set(auction_names) - set(matched['player_name_norm'].unique())

    # Names in enhanced not matched - we will attempt fuzzy choices for these if needed
    enh_unmatched = enh[~enh['player_name_norm'].isin(auction_names)].copy()

    print(f"[INFO] enhanced total rows: {len(enh)}")
    print(f"[INFO] exact matched rows: {exact_matched_count}")
    print(f"[INFO] unmatched enhanced rows (candidates for fuzzy): {len(enh_unmatched)}")
    print(f"[INFO] auction list size: {len(auction_names)}")

    # Fuzzy-match step: attempt to find suggestions for unmatched enhanced names
    fuzzy_matches = []
    fuzzy_matched_rows = []
    for i, row in enh_unmatched.iterrows():
        n = row['player_name_norm']
        if n == "":
            continue
        suggestions = find_fuzzy(n, auction_names)
        if suggestions:
            # pick the top suggestion as automatic mapping if you're comfortable — but for safety we only PRINT and collect
            fuzzy_matches.append((row['player_name'], n, suggestions))
            # Automatic mapping: uncomment next lines to accept top suggestion automatically.
            # top = suggestions[0]
            # mapped_rows = auction[auction['player_name_norm']==top]
            # if not mapped_rows.empty:
            #     fuzzy_matched_rows.append(row.to_dict())

    # Print a sample of fuzzy suggestions (max 50)
    print("\n[INFO] Fuzzy suggestions (sample up to 50). Format: (enhanced_name, norm, [suggestions])")
    for t in fuzzy_matches[:50]:
        print(t)

    # Ask user whether to auto-accept fuzzy matches? We'll be conservative: write only exact matches by default.
    # If you want to accept fuzzy suggestions automatically, set auto_accept=True below.
    auto_accept = False

    if auto_accept and fuzzy_matches:
        auto_count = 0
        for enh_name, norm, suggestions in fuzzy_matches:
            top = suggestions[0]
            # append all enhanced rows with this norm as matched to auction rows with top norm
            rows_to_add = enh[enh['player_name_norm']==norm]
            matched = pd.concat([matched, rows_to_add], ignore_index=True)
            auto_count += rows_to_add.shape[0]
        print(f"[INFO] Auto-accepted {auto_count} fuzzy matches")

    # Save exact-only matched file
    out = matched.drop_duplicates(subset=['player_id'])
    out.to_csv(OUT, index=False)
    print(f"[DONE] Saved filtered enhanced players (exact matches) -> {OUT}")
    print(f"[INFO] Rows saved: {out.shape[0]}")

    # Print unmatched top-20 enhanced players (helpful)
    print("\nSample unmatched enhanced players (first 30):")
    print(enh_unmatched[['player_id','player_name']].head(30).to_string(index=False))

    # Provide next-step hint
    print("\nNext steps:")
    print(f" - Inspect fuzzy suggestions above; to auto-accept fuzzy matches set auto_accept=True in this script.")
    print(" - If you want me to run fuzzy auto-mapping for you with a chosen cutoff, reply here and I will provide the script variant.")
    print(" - After filtering, run inference with the filtered players file using inference_catboost_v5.py (or the filtered inference script provided).")

if __name__ == '__main__':
    main()
