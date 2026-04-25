#!/usr/bin/env python3
"""
augment_batter_metrics.py (fixed)

Compute/fill missing avg_bat_pos and batting_hand into batter_metrics_ipl.csv using:
 - interim: player_match_stats_ipl_prepped.csv
 - players_master: players_master.csv or players.csv (fallback)

Saves updated file as batter_metrics_ipl_augmented.csv (keeps original).
"""
import os
from pathlib import Path
import pandas as pd

pd.options.mode.chained_assignment = None

# <-- CONFIG (change paths if needed) -->
processed_dir = Path(r"D:\RR_Hackathon_2025\data\processed")
interim_path = Path(r"D:\RR_Hackathon_2025\data\interim\player_match_stats_ipl_prepped.csv")
players_master_path = processed_dir.joinpath("players_master.csv")
players_csv_path = Path(r"D:\RR_Hackathon_2025\data\raw\players.csv")
batter_metrics_path = processed_dir.joinpath("batter_metrics_ipl.csv")
out_path = processed_dir.joinpath("batter_metrics_ipl_augmented.csv")
player_id_candidates = ['player_id','id','playerid','pid']
min_records_for_avgpos = 1

# ---------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_read(path):
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return None

# Load existing batter metrics
bat = safe_read(batter_metrics_path)
if bat is None:
    print(f"[ERROR] batter_metrics_ipl.csv not found at {batter_metrics_path}. Exiting.")
    raise SystemExit(1)

# Normalize player id col name in batter metrics
bat_id_col = find_col(bat, player_id_candidates)
if bat_id_col != 'player_id' and bat_id_col is not None:
    bat = bat.rename(columns={bat_id_col: 'player_id'})

# 1) Compute avg_bat_pos from interim file if missing or all NA
if 'avg_bat_pos' not in bat.columns or bat['avg_bat_pos'].isna().all():
    pm = safe_read(interim_path)
    if pm is None:
        print(f"[WARN] Interim file not found at {interim_path} — cannot compute avg_bat_pos automatically.")
    else:
        pm_id = find_col(pm, player_id_candidates)
        if pm_id is None:
            print("[WARN] No player id-like column in interim file.")
        else:
            # find batting position candidate columns
            batpos_col = find_col(pm, ['bat_pos','batting_position','bat_pos_in_innings','pos','batting_pos','position','batting_order'])
            if batpos_col:
                avgpos = pm.groupby(pm_id, as_index=False)[batpos_col].mean().rename(columns={batpos_col:'avg_bat_pos', pm_id:'player_id'})
                bat = bat.merge(avgpos, on='player_id', how='left')
                # if both existed, prefer existing values where not null
                if 'avg_bat_pos_x' in bat.columns and 'avg_bat_pos_y' in bat.columns:
                    bat['avg_bat_pos'] = bat['avg_bat_pos_x'].fillna(bat['avg_bat_pos_y'])
                    bat.drop(['avg_bat_pos_x','avg_bat_pos_y'], axis=1, inplace=True)
                print(f"[INFO] Computed avg_bat_pos from interim column '{batpos_col}'.")
            else:
                print("[WARN] No batting-position column found in interim — avg_bat_pos not computed.")

# 2) Fill batting_hand from players_master or players.csv (if missing)
need_fill_hand = ('batting_hand' not in bat.columns) or bat['batting_hand'].isna().all() or ((bat.get('batting_hand', pd.Series(['']*len(bat))).astype(str).str.strip()=='').all())
if need_fill_hand:
    players = safe_read(players_master_path)
    if players is None:
        players = safe_read(players_csv_path)

    if players is None:
        print("[WARN] players_master.csv and players.csv not found — cannot auto-fill batting_hand.")
    else:
        p_id = find_col(players, player_id_candidates)
        hand_col = find_col(players, ['batting_hand','bat_hand','bat_style','batting_style','batting'])
        if p_id is None:
            print("[WARN] No player id-like column in players file; cannot merge batting_hand.")
        elif hand_col is None:
            print("[WARN] No batting-hand-like column in players file to use.")
        else:
            players_sub = players[[p_id, hand_col]].rename(columns={p_id:'player_id', hand_col:'batting_hand'})
            def norm_hand(x):
                if pd.isna(x): return x
                xs = str(x).strip().upper()
                if xs == 'R' or xs.startswith('R'): return 'R'
                if xs == 'L' or xs.startswith('L'): return 'L'
                if 'RIGHT' in xs: return 'R'
                if 'LEFT' in xs: return 'L'
                return xs
            players_sub['batting_hand'] = players_sub['batting_hand'].apply(norm_hand)
            bat = bat.merge(players_sub, on='player_id', how='left')
            # handle merged columns safely
            if 'batting_hand_x' in bat.columns and 'batting_hand_y' in bat.columns:
                bat['batting_hand'] = bat['batting_hand_x'].fillna(bat['batting_hand_y'])
                bat.drop(['batting_hand_x','batting_hand_y'], axis=1, inplace=True)
            print("[INFO] Filled batting_hand from players file where available.")

# 3) Ensure ipl_balls exists (if not, try the 'balls' column or compute from interim)
if 'ipl_balls' not in bat.columns:
    if 'balls' in bat.columns:
        bat['ipl_balls'] = bat['balls']
        print("[INFO] Created ipl_balls from 'balls' column in batter_metrics.")
    else:
        pm = safe_read(interim_path)
        if pm is not None:
            pm_id = find_col(pm, player_id_candidates)
            balls_col = find_col(pm, ['balls','balls_faced','BF','b'])
            if pm_id and balls_col:
                balls_df = pm.groupby(pm_id, as_index=False)[balls_col].sum().rename(columns={pm_id:'player_id', balls_col:'ipl_balls'})
                bat = bat.merge(balls_df, on='player_id', how='left')
                bat['ipl_balls'] = bat['ipl_balls'].fillna(0).astype(int)
                print("[INFO] Computed ipl_balls by summing interim balls column.")
            else:
                bat['ipl_balls'] = 0
                print("[WARN] Could not compute ipl_balls from interim; set to 0.")
        else:
            bat['ipl_balls'] = 0
            print("[WARN] Interim not available; set ipl_balls to 0.")

# FINAL SAFETY: ensure batting_hand column exists before fillna and normalization
if 'batting_hand' not in bat.columns:
    bat['batting_hand'] = ''

# Normalize columns: dtype conversions and safe fill
if 'avg_bat_pos' in bat.columns:
    bat['avg_bat_pos'] = pd.to_numeric(bat['avg_bat_pos'], errors='coerce')
else:
    bat['avg_bat_pos'] = pd.NA

bat['batting_hand'] = bat['batting_hand'].fillna('').astype(str)
bat['ipl_balls'] = pd.to_numeric(bat['ipl_balls'], errors='coerce').fillna(0).astype(int)

# Save augmented file
bat.to_csv(out_path, index=False)
print(f"[INFO] Saved augmented batter metrics to {out_path}")
print("[INFO] Summary of key columns:")
print(" - avg_bat_pos: NA count =", bat['avg_bat_pos'].isna().sum(), " / total =", len(bat))
print(" - batting_hand: blank count =", (bat['batting_hand'].str.strip()=='' ).sum(), " / total =", len(bat))
print(" - ipl_balls: nonzero count =", (bat['ipl_balls']>0).sum(), " / total =", len(bat))

print("[NEXT] Re-run finish_day1.py (it will pick up batter_metrics_ipl.csv if you overwrite it, or use batter_metrics_ipl_augmented.csv by renaming).")
