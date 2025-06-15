
"""Run experiments with train/valid/test splits and proper WB-ML training.

Usage:
    python experiments/run_experiments.py --lambda_grid 0.0 0.1 0.3 1.0 3.0 --n_seeds 10
"""
import argparse, os, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.model import WBLogisticModel
from src.metrics import tpr_gap, dp_gap

def welfare_index(y_true, y_pred, gamma=1.0, rho=2):
    u = np.where(y_true == y_pred, gamma, 0.0)
    return np.sum(u ** (1 - rho) / (1 - rho))

def split_data(df, seed):
    """70-15-15 stratified on y."""
    X = df[[c for c in df.columns if c.startswith("x")]].values
    y = df["y"].values
    s = df["s"].values

    # primero train vs temp (valid+test)
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.30, random_state=seed, stratify=y
    )
    # luego valid vs test
    X_valid, X_test, y_valid, y_test, s_valid, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return (X_train, y_train, s_train,
            X_valid, y_valid, s_valid,
            X_test,  y_test,  s_test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_grid", nargs="+", type=float, required=True)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="data/synthetic_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    os.makedirs("results", exist_ok=True)
    records = []

    for seed in range(args.n_seeds):
        (X_tr, y_tr, s_tr,
         X_val, y_val, s_val,
         X_te, y_te, s_te) = split_data(df, seed)

        for lam in args.lambda_grid:
            # Si quieres búsqueda de C, aquí un placeholder
            model = WBLogisticModel(lambda_norm=lam, C=1.0)
            model.fit(X_tr, y_tr, s_tr)

            y_pred = model.predict(X_te)

            records.append({
                "lambda_norm": lam,
                "seed": seed,
                "Accuracy": accuracy_score(y_te, y_pred),
                "MacroF1": f1_score(y_te, y_pred, average="macro"),
                "TPR_gap": tpr_gap(y_te, y_pred, s_te),
                "DP_gap": dp_gap(y_pred, s_te),
                "Welfare": welfare_index(y_te, y_pred)
            })

    pd.DataFrame(records).to_csv("results/run_results.csv", index=False)
    print("✅ Resultados guardados en results/run_results.csv")

if __name__ == "__main__":
    main()
