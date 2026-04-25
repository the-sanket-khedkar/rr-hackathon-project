#!/usr/bin/env python3
"""
generate_visuals.py

Generates presentation-ready visuals and simplified CSV summaries from:
 - D:\RR_Hackathon_2025\data\processed\predictions_catboost_v5.csv
 - (optional) D:\RR_Hackathon_2025\data\processed\final_players_features.csv

Outputs saved to:
 - D:\RR_Hackathon_2025\data\processed\plots\*.png
 - D:\RR_Hackathon_2025\data\processed\top20_summary.csv
 - D:\RR_Hackathon_2025\data\processed\category_summary.csv
 - D:\RR_Hackathon_2025\data\processed\presentation_table.csv

Run:
    python generate_visuals.py
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (edit if different) ---
ROOT = Path(r"D:\RR_Hackathon_2025")
PROC = ROOT / "data" / "processed"
SCRIPTS = ROOT / "scripts"
PRED_PATH = PROC / "predictions_catboost_v5.csv"
PLAYERS_PATH = PROC / "final_players_features.csv"
OUT_DIR = PROC / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- load predictions ---
if not PRED_PATH.exists():
    raise FileNotFoundError(f"Predictions file not found at {PRED_PATH}. Run the model first.")
pred = pd.read_csv(PRED_PATH, low_memory=False)

# Optional: merge with player metadata for roles/nationality/age
if PLAYERS_PATH.exists():
    players = pd.read_csv(PLAYERS_PATH, low_memory=False)
    # normalize merge keys
    if 'player_id' in pred.columns and 'player_id' in players.columns:
        df = pred.merge(players[['player_id','player_name','avg_bat_pos','batting_type','bowling_type','nationality','date_of_birth','is_wicket_keeper']],
                        on='player_id', how='left', suffixes=('','_p'))
    else:
        # fallback to player_name join
        players['player_name_norm'] = players['player_name'].astype(str).str.strip().str.lower()
        pred['player_name_norm'] = pred['player_name'].astype(str).str.strip().str.lower()
        df = pred.merge(players[['player_name_norm','avg_bat_pos','batting_type','bowling_type','nationality','date_of_birth']],
                        on='player_name_norm', how='left', suffixes=('','_p'))
else:
    print("[WARN] players metadata not found; role / nationality charts will be limited.")
    df = pred.copy()

# compute age if date_of_birth present
if 'date_of_birth' in df.columns:
    try:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        df['age'] = df['date_of_birth'].dt.year.map(lambda y: 2025 - y if not pd.isna(y) else np.nan)
    except Exception:
        df['age'] = np.nan
else:
    df['age'] = np.nan

# create role buckets from avg_bat_pos and bowling_type
def role_from_row(row):
    # priorities: wicket keeper, batting roles, bowling-only
    if pd.notna(row.get('is_wicket_keeper')) and int(row.get('is_wicket_keeper',0))==1:
        return 'Wicketkeeper'
    abp = row.get('avg_bat_pos', np.nan)
    if pd.notna(abp) and abp>0:
        if abp <= 2.5:
            return 'Opener'
        if abp <= 6:
            return 'Middle-order'
        return 'Finisher'
    # fallback based on bowling_type
    bt = row.get('bowling_type', '')
    if pd.notna(bt) and str(bt).strip() != '':
        return 'Bowler'
    return 'All-rounder'

df['role'] = df.apply(role_from_row, axis=1)

# create a simple "likely_sold" flag threshold (p_sold > median)
median_psold = df['p_sold'].median()
df['likely_sold'] = (df['p_sold'] >= median_psold).astype(int)

# create presentation-friendly columns
df['price_if_sold_rounded_lakhs'] = (df['pred_price_if_sold'] / 1e5).round(2)
df['expected_price_lakhs'] = (df['expected_price'] / 1e5).round(2)

# ---- 1) Top 20 expected price bar chart ----
top20 = df.sort_values('expected_price', ascending=False).head(20).copy()
plt.figure(figsize=(10,8))
plt.barh(top20['player_name'].astype(str), top20['expected_price_lakhs'])
plt.xlabel("Expected price (lakhs ₹)")
plt.title("Top 20 Players by Expected Price")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
p1 = OUT_DIR / "top20_expected_price_lakhs.png"
plt.savefig(p1, dpi=200)
plt.close()

# ---- 2) Role-wise average expected price ----
role_stats = df.groupby('role').agg(
    count=('player_name','count'),
    mean_expected_lakhs=('expected_price_lakhs','mean'),
    median_expected_lakhs=('expected_price_lakhs','median'),
    mean_p_sold=('p_sold','mean')
).reset_index().sort_values('mean_expected_lakhs', ascending=False)

plt.figure(figsize=(8,5))
plt.bar(role_stats['role'], role_stats['mean_expected_lakhs'])
plt.title("Average expected price by role (lakhs ₹)")
plt.ylabel("Expected price (lakhs ₹)")
plt.xticks(rotation=30)
plt.tight_layout()
p2 = OUT_DIR / "role_avg_expected_price_lakhs.png"
plt.savefig(p2, dpi=200)
plt.close()

# ---- 3) Nationality distribution (top countries) ----
if 'nationality' in df.columns:
    nat_counts = df.groupby('nationality').agg(count=('player_name','count'),
                                              mean_expected=('expected_price_lakhs','mean')).reset_index()
    # display top 10 by count
    top_nations = nat_counts.sort_values('count', ascending=False).head(10)
    plt.figure(figsize=(10,5))
    plt.bar(top_nations['nationality'].astype(str), top_nations['mean_expected'])
    plt.title("Top 10 nationalities: mean expected price (lakhs ₹)")
    plt.ylabel("Mean expected price (lakhs ₹)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    p3 = OUT_DIR / "nationality_mean_expected_lakhs_top10.png"
    plt.savefig(p3, dpi=200)
    plt.close()
else:
    p3 = None

# ---- 4) Age vs expected price scatter ----
if df['age'].notna().sum() > 10:
    plt.figure(figsize=(8,5))
    plt.scatter(df['age'], df['expected_price_lakhs'], alpha=0.6)
    plt.xlabel("Age (years)")
    plt.ylabel("Expected price (lakhs ₹)")
    plt.title("Age vs Expected Price")
    plt.tight_layout()
    p4 = OUT_DIR / "age_vs_expected_price_lakhs.png"
    plt.savefig(p4, dpi=200)
    plt.close()
else:
    p4 = None

# ---- 5) p_sold distribution histogram ----
plt.figure(figsize=(8,4))
plt.hist(df['p_sold'], bins=30)
plt.xlabel("p_sold")
plt.title("Distribution of predicted probability of being sold (p_sold)")
plt.tight_layout()
p5 = OUT_DIR / "p_sold_distribution.png"
plt.savefig(p5, dpi=200)
plt.close()

# ---- 6) Price-if-sold discrete values histogram (in lakhs) ----
plt.figure(figsize=(8,4))
plt.hist(df['price_if_sold_rounded_lakhs'].dropna(), bins=40)
plt.xlabel("Price if sold (lakhs ₹)")
plt.title("Distribution of predicted price if sold")
plt.tight_layout()
p6 = OUT_DIR / "price_if_sold_distribution_lakhs.png"
plt.savefig(p6, dpi=200)
plt.close()

# ---- 7) SHAP placeholder (if shap png exists, copy to plots) ----
# If you already have a shap plot image at PROC/shap_reg_summary_fixed.png, copy it to OUT_DIR for the booklet.
shap_src = PROC / "shap_reg_summary_fixed.png"
if shap_src.exists():
    import shutil
    shutil.copy(shap_src, OUT_DIR / "shap_reg_summary_fixed.png")
    p_shap = OUT_DIR / "shap_reg_summary_fixed.png"
else:
    p_shap = None

# ---- Simplified CSVs for presentation ----
top20_table = top20[['player_id','player_name','p_sold','price_if_sold_rounded_lakhs','expected_price_lakhs','role']].copy()
top20_table = top20_table.rename(columns={
    'price_if_sold_rounded_lakhs': 'price_if_sold_lakhs',
    'expected_price_lakhs': 'expected_price_lakhs'
})
top20_csv = PROC / "top20_summary.csv"
top20_table.to_csv(top20_csv, index=False)

category_csv = PROC / "category_summary.csv"
role_stats.to_csv(category_csv, index=False)

# ---- Presentation table: top 100 with concise fields ----
presentation_cols = ['player_id','player_name','role','nationality','age','p_sold','price_if_sold_rounded_lakhs','expected_price_lakhs']
present_table = df[presentation_cols].sort_values('expected_price_lakhs', ascending=False).head(100)
present_table = present_table.rename(columns={
    'price_if_sold_rounded_lakhs': 'price_if_sold_lakhs',
    'expected_price_lakhs': 'expected_price_lakhs'
})
present_csv = PROC / "presentation_table.csv"
present_table.to_csv(present_csv, index=False)

# ---- Print summary of outputs ----
print("Saved plots to:", OUT_DIR)
print("Top20 CSV:", top20_csv)
print("Category summary CSV:", category_csv)
print("Presentation table CSV:", present_csv)
print("Plot files (samples):")
print(" -", p1)
print(" -", p2)
if p3: print(" -", p3)
if p4: print(" -", p4)
print(" -", p5)
print(" -", p6)
if p_shap: print(" -", p_shap)
print("\nYou're ready to include these images in the final PDF/slide deck.")
