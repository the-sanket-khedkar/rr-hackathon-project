#!/usr/bin/env python3
"""
finish_day1.py
Robust end-of-Day-1 finishing / feature assembly script for RR_Hackathon_2025.

Features:
- Defensive handling for missing 'ipl_balls' column (attempts many fallbacks).
- If batter_metrics_ipl.csv is missing, attempt to compute minimal batter metrics
  from player_match_stats_ipl_prepped.csv.
- Loads bowler metrics if present.
- Produces final_players_features.csv with at least the columns used downstream.
- Saves intermediate diagnostics for inspection.

Usage:
    python finish_day1.py
    python finish_day1.py --raw_interim path/to/player_match_stats_ipl_prepped.csv --processed_dir path/to/processed
"""

import os
import argparse
import pandas as pd
import sys
import warnings
from pathlib import Path

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------
# Helpers
# ---------------------------
def find_existing_column(df, candidates):
    """Return first candidate that exists in df.columns, otherwise None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
        return None
    except Exception as e:
        print(f"[WARN] Error reading {path}: {e}")
        return None

def compute_batter_metrics_from_interim(interim_path, id_col='player_id', save_path=None):
    """
    Compute minimal batter metrics (ipl_balls, runs, innings, avg_bat_pos) from an interim player-match csv.
    This is conservative: it tries a set of candidate column names and produces a small batter metrics DataFrame.
    """
    pm = safe_read_csv(interim_path, low_memory=False)
    if pm is None:
        print(f"[ERROR] Cannot compute batter metrics because interim file not found: {interim_path}")
        return None

    # Standardize some columns if present
    player_id_col = find_existing_column(pm, [id_col, 'id', 'playerid', 'player_id'])
    if player_id_col is None:
        raise ValueError("No player_id-like column found in interim file; cannot compute batter metrics")

    # Detect possible batting columns
    balls_col = find_existing_column(pm, ['balls','balls_faced','BF','b','balls_faced_innings'])
    runs_col = find_existing_column(pm, ['runs','batted_runs','bat_runs','R','runs_scored'])
    bat_pos_col = find_existing_column(pm, ['bat_pos','batting_position','pos','batting_pos','bat_pos_avg','batting_order'])

    # Make copies of the columns we can use
    use_cols = [player_id_col]
    if balls_col: use_cols.append(balls_col)
    if runs_col: use_cols.append(runs_col)
    if bat_pos_col: use_cols.append(bat_pos_col)

    pm_small = pm[use_cols].copy()

    # Compute aggregates
    agg_dict = {}
    if balls_col:
        agg_dict[balls_col] = 'sum'
    if runs_col:
        agg_dict[runs_col] = 'sum'
    if bat_pos_col:
        agg_dict[bat_pos_col] = 'mean'

    grouped = pm_small.groupby(player_id_col, as_index=False).agg(agg_dict)
    # Normalize names to expected ones
    rename_map = {}
    if balls_col:
        rename_map[balls_col] = 'ipl_balls'
    if runs_col:
        rename_map[runs_col] = 'ipl_runs'
    if bat_pos_col:
        rename_map[bat_pos_col] = 'avg_bat_pos'

    grouped = grouped.rename(columns=rename_map)

    # Add safe defaults for missing metrics
    if 'ipl_balls' not in grouped.columns:
        grouped['ipl_balls'] = 0
    if 'ipl_runs' not in grouped.columns:
        grouped['ipl_runs'] = 0
    if 'avg_bat_pos' not in grouped.columns:
        grouped['avg_bat_pos'] = pd.NA

    # Ensure player id column name consistency
    if player_id_col != id_col:
        grouped = grouped.rename(columns={player_id_col: id_col})

    if save_path:
        grouped.to_csv(save_path, index=False)
        print(f"[INFO] Saved computed batter metrics to {save_path}")

    return grouped

def ensure_ipl_balls(bat_df, batter_metrics_path=None, interim_path=None, id_col='player_id'):
    """
    Ensure 'ipl_balls' column exists in bat_df.
    Strategies:
      1) If present, return unchanged.
      2) If alternative names exist in bat_df, create ipl_balls from them.
      3) Try merging 'ipl_balls' from batter_metrics_path if present.
      4) Try computing from interim_path (player_match_stats prepped).
      5) Final fallback: create zero-filled 'ipl_balls' and warn.
    """
    df = bat_df.copy()
    if 'ipl_balls' in df.columns:
        return df

    # Try alt names in bat_df
    alt_names = ['balls','balls_faced','ipl_balls_faced','BF','ipl_bf']
    for alt in alt_names:
        if alt in df.columns:
            df['ipl_balls'] = df[alt]
            print(f"[INFO] Created 'ipl_balls' from existing column '{alt}' in batter df.")
            return df

    # Try merging from batter_metrics_path
    if batter_metrics_path and os.path.exists(batter_metrics_path):
        bm = safe_read_csv(batter_metrics_path)
        if bm is not None:
            bm_col = find_existing_column(bm, ['ipl_balls','balls','balls_faced','BF','ipl_balls_faced'])
            if bm_col:
                if id_col not in bm.columns:
                    # attempt to find id column in bm
                    bm_id = find_existing_column(bm, [id_col, 'id','player_id','playerid'])
                    if bm_id:
                        bm = bm.rename(columns={bm_id: id_col})
                if id_col in bm.columns:
                    bm_sub = bm[[id_col, bm_col]].rename(columns={bm_col: 'ipl_balls'})
                    df = df.merge(bm_sub, on=id_col, how='left')
                    if 'ipl_balls' not in df.columns:
                        df['ipl_balls'] = 0
                    df['ipl_balls'] = df['ipl_balls'].fillna(0).astype(int)
                    print(f"[INFO] Merged 'ipl_balls' from {batter_metrics_path} (col: {bm_col})")
                    return df
                else:
                    print(f"[WARN] batter_metrics file {batter_metrics_path} has no id-like column to merge on.")
            else:
                print(f"[WARN] No balls-like column found in {batter_metrics_path}.")
        else:
            print(f"[WARN] Could not read {batter_metrics_path} to extract ipl_balls.")

    # Try compute from interim_path
    if interim_path and os.path.exists(interim_path):
        print(f"[INFO] Attempting to compute ipl_balls from interim file: {interim_path}")
        bm_calc = compute_batter_metrics_from_interim(interim_path, id_col=id_col)
        if bm_calc is not None and 'ipl_balls' in bm_calc.columns:
            df = df.merge(bm_calc[[id_col,'ipl_balls']], on=id_col, how='left')
            df['ipl_balls'] = df['ipl_balls'].fillna(0).astype(int)
            print(f"[INFO] Computed and merged 'ipl_balls' from interim file.")
            return df
        else:
            print(f"[WARN] Failed to compute ipl_balls from interim file {interim_path}.")

    # Final fallback
    print("[WARN] 'ipl_balls' not found by any strategy — creating fallback column with zeros.")
    df['ipl_balls'] = 0
    return df

# ---------------------------
# Main
# ---------------------------
def main(args):
    # Paths (defaults matching your environment; override via CLI)
    processed_dir = Path(args.processed_dir)
    interim_file = Path(args.interim_file)
    players_master = processed_dir.joinpath('players_master.csv')
    batter_metrics_path = processed_dir.joinpath('batter_metrics_ipl.csv')
    bowler_metrics_path = processed_dir.joinpath('bowler_metrics_ipl.csv')
    output_features = processed_dir.joinpath('final_players_features.csv')

    # Read players master (if exists)
    players_df = safe_read_csv(players_master) if players_master.exists() else None
    if players_df is None:
        print(f"[WARN] players_master.csv not found at {players_master}. Proceeding without it.")

    # Read or compute batter metrics
    if batter_metrics_path.exists():
        print(f"[INFO] Found existing batter metrics at {batter_metrics_path}; loading.")
        bat = safe_read_csv(batter_metrics_path, low_memory=False)
        if bat is None:
            print(f"[WARN] Unexpected error reading {batter_metrics_path}; attempting to compute from interim.")
            bat = compute_batter_metrics_from_interim(str(interim_file), id_col='player_id', save_path=None)
    else:
        print(f"[INFO] batter_metrics_ipl.csv not found at {batter_metrics_path}; computing from interim file.")
        bat = compute_batter_metrics_from_interim(str(interim_file), id_col='player_id', save_path=str(batter_metrics_path))
        if bat is None:
            print("[ERROR] Could not obtain batter metrics. Exiting.")
            sys.exit(1)

    # Ensure common columns exist
    if 'player_id' not in bat.columns:
        alt_id = find_existing_column(bat, ['id','playerid','player_id'])
        if alt_id:
            bat = bat.rename(columns={alt_id: 'player_id'})
        else:
            raise ValueError("batter metrics have no player id column; cannot proceed")

    # Ensure ipl_balls column exists (this prevents the KeyError you hit)
    bat = ensure_ipl_balls(bat, batter_metrics_path=str(batter_metrics_path) if batter_metrics_path.exists() else None,
                           interim_path=str(interim_file) if interim_file.exists() else None,
                           id_col='player_id')

    # Normalize avg_bat_pos presence
    if 'avg_bat_pos' not in bat.columns:
        # look for common alternatives and map them
        alt = find_existing_column(bat, ['mean_bat_pos','avg_batting_position','bat_pos','batting_pos'])
        if alt:
            bat = bat.rename(columns={alt: 'avg_bat_pos'})
            print(f"[INFO] Renamed {alt} -> avg_bat_pos in batter metrics.")
        else:
            bat['avg_bat_pos'] = pd.NA
            print("[WARN] avg_bat_pos not found; filled with NA.")

    # Normalize batting_hand
    if 'batting_hand' not in bat.columns:
        alt = find_existing_column(bat, ['bat_hand','batting_style','hand'])
        if alt:
            bat = bat.rename(columns={alt: 'batting_hand'})
            print(f"[INFO] Renamed {alt} -> batting_hand.")
        else:
            # If we don't have, create NA column so the selection expression won't crash on .str.upper()
            bat['batting_hand'] = pd.NA
            print("[WARN] batting_hand not found; filled with NA.")

    # Safely compute 'mid' selection
    min_balls = args.min_balls
    # Ensure numeric types where needed
    # convert avg_bat_pos to numeric if possible
    try:
        bat['avg_bat_pos'] = pd.to_numeric(bat['avg_bat_pos'], errors='coerce')
    except Exception:
        pass

    # For batting_hand string operations, fillna with empty string
    bat['batting_hand'] = bat['batting_hand'].fillna('').astype(str)

    # Apply selection safely: check for presence of required columns
    # Condition: (bat['avg_bat_pos'].between(3,6)) & (bat['batting_hand'].str.upper()=='R') & (bat['ipl_balls']>=min_balls)
    cond_avgpos = bat['avg_bat_pos'].between(3,6) if 'avg_bat_pos' in bat.columns else pd.Series([False]*len(bat))
    cond_hand = bat['batting_hand'].str.upper() == 'R'
    cond_balls = pd.to_numeric(bat['ipl_balls'], errors='coerce').fillna(0).astype(int) >= int(min_balls)

    mid_mask = cond_avgpos & cond_hand & cond_balls
    mid = bat[mid_mask].copy()
    print(f"[INFO] Selected {len(mid)} 'mid' players with min_balls={min_balls}.")

    # Optional: produce 'openers' and 'finishers' buckets for convenience
    openers_mask = (bat['avg_bat_pos'] <= 2) & (pd.to_numeric(bat['ipl_balls'], errors='coerce').fillna(0).astype(int) >= int(min_balls))
    openers = bat[openers_mask].copy()
    finishers_mask = (bat['avg_bat_pos'] >= 7) & (pd.to_numeric(bat['ipl_balls'], errors='coerce').fillna(0).astype(int) >= int(min_balls))
    finishers = bat[finishers_mask].copy()

    print(f"[INFO] Openers selected: {len(openers)}, Finishers selected: {len(finishers)}")

    # Read bowler metrics if available (you previously saved it)
    bow = safe_read_csv(bowler_metrics_path) if bowler_metrics_path.exists() else None
    if bow is None:
        print(f"[WARN] bowler_metrics_ipl.csv not found at {bowler_metrics_path}. Proceeding without bowler metrics.")
    else:
        # Ensure player_id col exists in bow too
        if 'player_id' not in bow.columns:
            alt = find_existing_column(bow, ['id','playerid','player_id'])
            if alt:
                bow = bow.rename(columns={alt: 'player_id'})

    # Build final features DataFrame by merging players_master (if exists), bat, bow
    final = bat.copy()
    if players_df is not None:
        # ensure players_df has player_id
        if 'player_id' not in players_df.columns:
            alt = find_existing_column(players_df, ['id','playerid','player_id'])
            if alt:
                players_df = players_df.rename(columns={alt: 'player_id'})
        # left merge so we keep only players present in bat
        final = final.merge(players_df, on='player_id', how='left', suffixes=('','_players'))
        print(f"[INFO] Merged players_master into final features; resulting shape: {final.shape}")

    if bow is not None:
        final = final.merge(bow, on='player_id', how='left', suffixes=('','_bow'))
        print(f"[INFO] Merged bowler metrics into final features; resulting shape: {final.shape}")

    # Save final features
    try:
        final.to_csv(output_features, index=False)
        print(f"[INFO] Saved final features to {output_features} (rows: {len(final)}, cols: {len(final.columns)})")
    except Exception as e:
        print(f"[ERROR] Failed to save {output_features}: {e}")

    # Also save the buckets for convenience
    try:
        mid.to_csv(processed_dir.joinpath('mid_players.csv'), index=False)
        openers.to_csv(processed_dir.joinpath('openers_players.csv'), index=False)
        finishers.to_csv(processed_dir.joinpath('finishers_players.csv'), index=False)
        print(f"[INFO] Saved mid/openers/finishers CSVs to {processed_dir}")
    except Exception as e:
        print(f"[WARN] Failed to save category CSVs: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finish Day1 preprocessing — robust version.")
    parser.add_argument('--processed_dir', default=r"D:\RR_Hackathon_2025\data\processed", help="Processed data dir")
    parser.add_argument('--interim_file', default=r"D:\RR_Hackathon_2025\data\interim\player_match_stats_ipl_prepped.csv", help="Interim per-match player stats (prepped)")
    parser.add_argument('--min_balls', default=50, type=int, help="Minimum balls threshold for selection")
    args = parser.parse_args()
    main(args)
