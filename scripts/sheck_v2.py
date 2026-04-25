import pandas as pd, numpy as np, joblib
proc = r"D:\RR_Hackathon_2025\data\processed"
models = r"D:\RR_Hackathon_2025\models"

train = pd.read_csv(f"{proc}\\model_data.csv", low_memory=False)
players = pd.read_csv(f"{proc}\\final_players_features.csv", low_memory=False)
clf = joblib.load(f"{models}\\lgbm_sold_classifier.pkl")
reg = joblib.load(f"{models}\\lgbm_price_regressor.pkl")

# reconstruct feature cols (same as inference script)
exclude = {'player_id','player_name','sold_price','sold','price','log_price'}
feature_cols = [c for c in train.columns if c not in exclude]

# Prepare X exactly like inference_v2 did (you can reuse code)
# ... here we'll just re-use predictions file
preds = pd.read_csv(f"{proc}\\predictions_v2.csv")

# Diagnostics
print("Model train log_price stats:")
print(train['log_price'].describe())

print("\nPredicted log_price (inference) summary:")
# We don't have direct log preds in predictions_v2; compute by inverting expm1
pred_price_if_sold = preds['pred_price_if_sold'].values
# protect zeros
with np.errstate(divide='ignore'):
    pred_log = np.log1p(pred_price_if_sold)
print(pd.Series(pred_log).describe())

print("\nPredicted price if sold (head):")
print(pred_price_if_sold[:10])
