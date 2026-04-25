# scripts/preprocess.py
import os
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
INTERIM = os.path.join(BASE, "data", "interim")
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(INTERIM, exist_ok=True)

def safe_read(path):
    if not os.path.exists(path):
        print("MISSING:", path)
        return None
    try:
        if path.lower().endswith(('.xls','.xlsx')):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception as e:
        print("READ ERROR", path, e)
        return None

def build_players_master():
    p = safe_read(os.path.join(RAW, "players.csv"))
    if p is None:
        return
    p['player_name'] = p['player_name'].astype(str).str.strip()
    if 'player_id' not in p.columns:
        p['player_id'] = p['player_name'].apply(lambda x: abs(hash(x)) % 10**9)
    outp = os.path.join(PROCESSED, "players_master.csv")
    p.to_csv(outp, index=False)
    print("Saved", outp)

def build_aggregates():
    pms = safe_read(os.path.join(RAW, "player_match_stats.csv"))
    if pms is None:
        print("player_match_stats not found, abort.")
        return
    # minimal cleaning example
    pms.columns = [c.strip() for c in pms.columns]
    pms['runs_scored'] = pd.to_numeric(pms.get('runs_scored', 0), errors='coerce').fillna(0)
    pms['balls_faced'] = pd.to_numeric(pms.get('balls_faced', 0), errors='coerce').fillna(0)
    # batter aggregates
    bat = pms.groupby(['player_id','player_name']).agg(
        matches=('match_id','nunique'),
        runs=('runs_scored','sum'),
        balls=('balls_faced','sum')
    ).reset_index()
    bat['sr'] = (bat['runs'] / bat['balls'] * 100).replace([pd.NA, np.inf], pd.NA)
    bat.to_csv(os.path.join(PROCESSED, "batter_metrics_ipl.csv"), index=False)
    print("Saved batter_metrics_ipl.csv")

if __name__ == "__main__":
    print("Running Day-1 preprocess skeleton...")
    build_players_master()
    build_aggregates()
    print("Done.")
