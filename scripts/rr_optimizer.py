
import pandas as pd
import argparse

def optimize(input_path, budget, output_path):
    df = pd.read_csv(input_path)
    if 'expected_price' not in df.columns:
        raise ValueError("expected_price column missing")

    df_sorted = df.sort_values('expected_price', ascending=False)

    picks = []
    spent = 0.0

    for _, row in df_sorted.iterrows():
        price = row['corrected_pred_price_if_sold'] if 'corrected_pred_price_if_sold' in df.columns else row['expected_price']
        if spent + price <= budget:
            picks.append(row)
            spent += price
        if spent >= budget:
            break

    out = pd.DataFrame(picks)
    out.to_csv(output_path, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    optimize(args.input, args.budget, args.out)
    print("[DONE] Saved:", args.out)
