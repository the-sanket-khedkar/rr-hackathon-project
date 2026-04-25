# run_shap_analysis_fixed.py
import joblib, pandas as pd, numpy as np, json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = Path(r"D:\RR_Hackathon_2025\models")
PROC = Path(r"D:\RR_Hackathon_2025\data\processed")

# paths
clf_path = MODELS / "lgbm_sold_classifier_v3.pkl"
reg_path = MODELS / "lgbm_price_regressor_v3.pkl"
train_path = PROC / "model_data.csv"
cat_maps_path = MODELS / "cat_maps.json"

# load
clf = joblib.load(clf_path)
reg = joblib.load(reg_path)
train = pd.read_csv(train_path, low_memory=False)

# reconstruct feature list (same as earlier)
exclude = {'player_id','player_name','sold_price','sold','price','log_price','log_price_lakhs','price_lakhs',
           'player_name_players','player_name_bow'}
feature_cols = [c for c in train.columns if c not in exclude]
print("Feature count:", len(feature_cols))

# load cat maps (produced at train time)
with open(cat_maps_path, 'r', encoding='utf8') as f:
    cat_maps = json.load(f)

# prepare X (from train DF to avoid divergent inference transforms)
X = train[feature_cols].copy()

# --- Fix categorical dtypes: map using cat_maps where available ---
def apply_train_map(series, mapping_list):
    # mapping_list is list of categories in training order
    mp = {cat: i for i, cat in enumerate(mapping_list)}
    default = 0 if len(mapping_list)>0 else -1
    # treat NaN/'' as 'NA' mapping if present in map
    def _map(v):
        if pd.isna(v) or str(v)=='':
            v = "NA"
        v = str(v)
        return mp[v] if v in mp else default
    return series.fillna("NA").astype(str).map(_map).astype(int)

# Map all categorical columns that we have maps for and that exist in feature_cols
for c, mapping_list in cat_maps.items():
    if c in feature_cols and c in X.columns:
        X[c] = apply_train_map(X[c], mapping_list)

# Convert date_of_birth -> numeric year (if present)
if 'date_of_birth' in X.columns:
    try:
        X['date_of_birth'] = pd.to_datetime(X['date_of_birth'], errors='coerce')
        X['date_of_birth'] = X['date_of_birth'].dt.year.fillna(0).astype(int)
    except Exception:
        # fallback: try numeric coercion
        X['date_of_birth'] = pd.to_numeric(X['date_of_birth'], errors='coerce').fillna(0).astype(int)

# Convert any remaining object columns to numeric where possible, or factorize them
for c in X.columns:
    if X[c].dtype == object:
        # attempt numeric coercion first
        coerced = pd.to_numeric(X[c], errors='coerce')
        if coerced.notna().sum() > 0:  # some numeric values
            X[c] = coerced.fillna(0)
        else:
            # final fallback: factorize (preserves deterministic numeric)
            X[c], _ = pd.factorize(X[c].astype(str))
    # ensure no NaNs
    if X[c].isna().any():
        X[c] = X[c].fillna(0)

# final dtype check
bad = [c for c in X.columns if X[c].dtype not in [np.float64, np.float32, np.int64, np.int32, 'float64','int64']]
if bad:
    print("Warning - still bad dtypes:", bad)

# show top importances (existing)
print("Top classifier importances:")
print(sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: x[1], reverse=True)[:20])
print("\nTop regressor importances:")
print(sorted(zip(feature_cols, reg.feature_importances_), key=lambda x: x[1], reverse=True)[:20])

# Run SHAP for regressor (sample to keep it fast)
explainer = shap.TreeExplainer(reg)
sample = X.sample(min(500, len(X)), random_state=42)
print("Running SHAP on sample of size", len(sample))
sv = explainer.shap_values(sample)   # will now accept numeric input
plt.figure(figsize=(10,6))
shap.summary_plot(sv, sample, show=False)
out_png = PROC / "shap_reg_summary_fixed.png"
plt.title("SHAP summary (regressor) - fixed inputs")
plt.savefig(out_png, bbox_inches='tight', dpi=150)
print("Saved SHAP plot ->", out_png)
