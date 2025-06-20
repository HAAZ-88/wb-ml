
"""Run experiments with train/valid/test splits and WB‑ML (C=0.2).
Usage:
    python experiments/run_experiments.py --lambda_grid 0 0.5 1 2 5 10 --n_seeds 10
"""
import argparse, os, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import WBLogisticModel
from src.metrics import tpr_gap, dp_gap

def welfare_index(y_true, y_pred, gamma=1.0, rho=2):
    u = np.where(y_true == y_pred, gamma, 1e-6)
    return np.sum(u ** (1 - rho) / (1 - rho))

def split70_15_15(df, seed):
    X = df[[c for c in df.columns if c.startswith("x")]].values
    y = df["y"].values
    s = df["s"].values
    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
        X, y, s, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te, s_val, s_te = train_test_split(
        X_tmp, y_tmp, s_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    return (X_tr, y_tr, s_tr, X_val, y_val, s_val, X_te, y_te, s_te)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_grid", nargs="+", type=float,
                        default=[0, 0.5, 1, 2, 5, 10])
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--data_path", type=str,
                        default="data/synthetic_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in range(args.n_seeds):
        X_tr, y_tr, s_tr, _, _, _, X_te, y_te, s_te = split70_15_15(df, seed)
        for lam in args.lambda_grid:
            model = WBLogisticModel(lambda_norm=lam, C=0.2)  # lower C
            model.fit(X_tr, y_tr, s_tr)
            y_pred = model.predict(X_te)
            rows.append({
                "lambda_norm": lam,
                "seed": seed,
                "Accuracy": accuracy_score(y_te, y_pred),
                "MacroF1": f1_score(y_te, y_pred, average="macro"),
                "TPR_gap": tpr_gap(y_te, y_pred, s_te),
                "DP_gap": dp_gap(y_pred, s_te),
                "Welfare": welfare_index(y_te, y_pred)
            })
    pd.DataFrame(rows).to_csv("results/run_results_synthetic.csv", index=False)
    print("✅ results/run_results.csv written")

if __name__ == "__main__":
    main()
