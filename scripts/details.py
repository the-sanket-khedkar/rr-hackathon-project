# details.py (robust, per-player aggregation + name normalization)
import pandas as pd
from pathlib import Path

RAW = Path(r"D:\RR_Hackathon_2025\data\raw")
PROC = Path(r"D:\RR_Hackathon_2025\data\processed")

# Load
bids = pd.read_csv(RAW / "bid_details.csv", low_memory=False)
final = pd.read_csv(PROC / "final_players_features.csv", low_memory=False)

print("bid_details columns:", list(bids.columns))
print("final_players_features columns:", list(final.columns))

# Normalize player_name in both DFs to reduce mismatch noise
def norm_name(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

bids['player_name_norm'] = bids['player_name'].apply(norm_name)
final['player_name_norm'] = final['player_name'].apply(norm_name)

# Build sold flag
bids['sold'] = bids['status'].astype(str).str.upper().eq("SOLD").astype(int)

# Keep only rows that are useful (drop rows with empty normalized name)
bids = bids[bids['player_name_norm'] != ""]

# Option A: aggregate per player_name_norm
# - sold_any: 1 if any sold
# - price: max bid_amount among sold rows (fallback to 0)
sold_rows = bids[bids['sold'] == 1].copy()
agg_price = (
    sold_rows.groupby('player_name_norm', as_index=False)['bid_amount']
    .max()
    .rename(columns={'bid_amount': 'price'})
)

agg_sold_any = (
    bids.groupby('player_name_norm', as_index=False)['sold']
    .max()
    .rename(columns={'sold': 'sold_any'})
)

model_df = agg_sold_any.merge(agg_price, on='player_name_norm', how='left')
model_df['price'] = model_df['price'].fillna(0)
model_df = model_df.rename(columns={'sold_any': 'sold'})

print("\nPer-player model_df built. shape:", model_df.shape)
print("Sold counts (per-player):\n", model_df['sold'].value_counts())
print("Price stats (sold only):\n", model_df.loc[model_df['sold']==1, 'price'].describe())

# Merge with final on normalized name
merged = final.merge(model_df[['player_name_norm','sold','price']], on='player_name_norm', how='left', indicator=True)

print("\nMerge indicator counts:")
print(merged['_merge'].value_counts())

# For players without any bids in raw data, set sold=0 and price=0
merged['sold'] = merged['sold'].fillna(0).astype(int)
merged['price'] = merged['price'].fillna(0.0)

# Drop helper norm column before saving; keep player_name & player_id etc.
out_cols = list(final.columns) + ['sold','price']
# ensure we only keep columns that exist
out_cols = [c for c in out_cols if c in merged.columns]

out = merged[out_cols].copy()

out_path = PROC / "model_data.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved model dataset -> {out_path}")
print("Merged shape:", out.shape)
print("Sample unmatched players (first 10):")
print(merged[merged['_merge']=='left_only'][['player_name','player_name_norm']].head(10).to_string(index=False))
