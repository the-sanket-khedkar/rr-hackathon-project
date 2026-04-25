#!/usr/bin/env python3
"""
augment_player_features.py

Produces enhanced_players_features.csv with extra features:
 - recent form (last N innings runs, sr, avg)
 - last N matches bowling economy / wickets
 - phase SRs: powerplay (0-6), middle (7-15), death (16-20) for batting
 - boundary_pct, dot_ball_pct (batting)
 - role flags: is_opener / is_finisher / is_middle based on avg_bat_pos
 - age (if date_of_birth present)
 - career aggregates: career_sr, career_avg, career_balls, career_matches
Output: D:\RR_Hackathon_2025\data\processed\enhanced_players_features.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(r"D:\RR_Hackathon_2025\data\raw")
PROC = Path(r"D:\RR_Hackathon_2025\data\processed")

# Files we expect
player_matches_path = RAW / "player_match_stats.csv"
players_path = PROC / "final_players_features.csv"

# Load
pm = pd.read_csv(player_matches_path, low_memory=False)
players = pd.read_csv(players_path, low_memory=False)

# Normalize player_id and player_name assumptions
# pm expected columns: player_id, player_name, runs, balls, fours, sixes, overs, runs_conceded, wickets, dot_balls, batting_position, match_date (if available)
# we will be defensive: use columns if present

def safe_col(df, col, default=0):
    return df[col] if col in df.columns else pd.Series([default]*len(df))

# Convert match_date to datetime if present
if 'match_date' in pm.columns:
    pm['match_date'] = pd.to_datetime(pm['match_date'], errors='coerce')

# Ensure numeric columns exist
numeric_cols = ['runs','balls','fours','sixes','overs','runs_conceded','wickets','dot_balls','batting_position']
for c in numeric_cols:
    if c not in pm.columns:
        pm[c] = 0

# Create per-player rolling / recent stats (last N innings)
N_recent = 10
pm = pm.sort_values(['player_id','match_date']).reset_index(drop=True)

# Basic career aggregates
career = pm.groupby('player_id').agg(
    career_matches = ('match_id','nunique') if 'match_id' in pm.columns else ('runs','count'),
    career_runs = ('runs','sum'),
    career_balls = ('balls','sum'),
    career_fours = ('fours','sum'),
    career_sixes = ('sixes','sum'),
    career_wickets = ('wickets','sum'),
    career_dot_balls = ('dot_balls','sum'),
).reset_index()

career['career_sr'] = (career['career_runs'] / career['career_balls']).replace([np.inf, np.nan],0) * 100
career['career_avg'] = (career['career_runs'] / (career['career_matches'].replace(0,np.nan))).fillna(0)

# Last-N innings stats
def last_n_stats(df, n=10):
    # returns a df indexed by player_id with columns: lastN_runs_sum, lastN_sr, lastN_avg, lastN_balls_sum
    records=[]
    for pid, g in df.groupby('player_id'):
        g = g.sort_values('match_date') if 'match_date' in g.columns else g
        last = g.tail(n)
        runs = last['runs'].sum()
        balls = last['balls'].sum()
        avg = last['runs'].sum() / max((last['runs']!=0).sum(),1) if len(last)>0 else 0
        sr = (runs/balls*100) if balls>0 else 0
        records.append((pid, runs, balls, avg, sr))
    return pd.DataFrame(records, columns=['player_id','last{}_runs'.format(n),'last{}_balls'.format(n),'last{}_avg'.format(n),'last{}_sr'.format(n)])

lastN = last_n_stats(pm, N_recent)

# Phase-wise batting SRs (if ball-by-ball or ball-range columns not present, we approximate by batting_position)
# If pm has 'inning_over' or 'ball_in_inning' we could calculate exact phases; fallback: estimate by ball range not possible.
# We'll calculate approximations: powerplay if batting_position <= 3 and death if batting_position>=8 etc.
pm['is_pp'] = pm['batting_position'].apply(lambda x: 1 if (x>0 and x<=3) else 0)
pm['is_death'] = pm['batting_position'].apply(lambda x: 1 if (x>=8) else 0)
phase = pm.groupby('player_id').agg(
    pp_runs = ('runs', lambda s: s[pm.loc[s.index,'is_pp']==1].sum() if len(s)>0 else 0),
    pp_balls = ('balls', lambda s: s[pm.loc[s.index,'is_pp']==1].sum() if len(s)>0 else 0),
    death_runs = ('runs', lambda s: s[pm.loc[s.index,'is_death']==1].sum() if len(s)>0 else 0),
    death_balls = ('balls', lambda s: s[pm.loc[s.index,'is_death']==1].sum() if len(s)>0 else 0),
).reset_index()
phase['pp_sr'] = phase.apply(lambda r: (r['pp_runs']/r['pp_balls']*100) if r['pp_balls']>0 else 0, axis=1)
phase['death_sr'] = phase.apply(lambda r: (r['death_runs']/r['death_balls']*100) if r['death_balls']>0 else 0, axis=1)

# Boundary pct and dot ball pct (batting)
agg_batting = pm.groupby('player_id').agg(
    runs_total = ('runs','sum'),
    balls_total = ('balls','sum'),
    fours_total = ('fours','sum'),
    sixes_total = ('sixes','sum'),
    dot_balls_total = ('dot_balls','sum')
).reset_index()

agg_batting['boundary_pct'] = ((agg_batting['fours_total']*4 + agg_batting['sixes_total']*6) / agg_batting['runs_total']).replace([np.inf, np.nan],0)
agg_batting['dot_ball_pct'] = (agg_batting['dot_balls_total'] / agg_batting['balls_total']).replace([np.inf, np.nan],0)

# Bowling aggregates
agg_bowling = pm.groupby('player_id').agg(
    overs_total = ('overs','sum'),
    runs_conceded = ('runs_conceded','sum'),
    wickets_total = ('wickets','sum'),
    dot_balls_bow = ('dot_balls','sum')
).reset_index()
agg_bowling['bowling_economy'] = agg_bowling.apply(lambda r: (r['runs_conceded']/(r['overs_total']+1e-9)) if r['overs_total']>0 else 0, axis=1)
agg_bowling['bowling_sr'] = agg_bowling.apply(lambda r: (r['overs_total']*6 / (r['wickets_total']+1e-9)) if r['wickets_total']>0 else 0, axis=1)
agg_bowling['bow_dot_pct'] = (agg_bowling['dot_balls_bow'] / (agg_bowling['overs_total']*6 + 1e-9)).replace([np.inf, np.nan],0)

# role flags from avg_bat_pos in final players frame
players['is_opener'] = players['avg_bat_pos'].apply(lambda x: 1 if pd.notna(x) and x<=2.5 else 0)
players['is_finisher'] = players['avg_bat_pos'].apply(lambda x: 1 if pd.notna(x) and x>=6 else 0)
players['is_middle'] = players['avg_bat_pos'].apply(lambda x: 1 if pd.notna(x) and 2.5 < x < 6 else 0)

# age if dob present
if 'date_of_birth' in players.columns:
    players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], errors='coerce')
    ref_date = pd.Timestamp('2025-01-01')
    players['age'] = players['date_of_birth'].apply(lambda d: (ref_date.year - d.year) if pd.notna(d) else np.nan)
else:
    players['age'] = np.nan

# Merge all aggregates into players
out = players.copy()
out = out.merge(career, on='player_id', how='left')
out = out.merge(lastN, on='player_id', how='left')
out = out.merge(phase[['player_id','pp_sr','death_sr']], on='player_id', how='left')
out = out.merge(agg_batting[['player_id','boundary_pct','dot_ball_pct']], on='player_id', how='left')
out = out.merge(agg_bowling[['player_id','bowling_economy','bowling_sr','bow_dot_pct']], on='player_id', how='left')

# Fillna and basic cleanup
out = out.fillna(0)

# Save enhanced file
out_path = PROC / "enhanced_players_features.csv"
out.to_csv(out_path, index=False)
print("Saved enhanced features to:", out_path)
print("Columns:", out.columns.tolist())
