
"""Run experiments with train/valid/test splits and proper WB-ML training.
Usage:
    python experiments/run_experiments.py --data_path data/findex_bolivia_2021_processed.csv --lambda_grid 0 0.3 0.6 1 1.5 2 --n_seeds 10
"""
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import WBLogisticModel
from src.metrics import tpr_gap, dp_gap

def welfare_index(y_true, y_pred, gamma=1.0, rho=2):
    u = np.where(y_true == y_pred, gamma, 1e-6)  
    return np.sum(u ** (1 - rho) / (1 - rho))

def split_data(df, seed):
    # Selección explícita de features para datos Findex Bolivia
    feature_cols = ["educ", "age", "inc_q", "emp_in", "mobileowner", "internetaccess"]
    X = df[feature_cols].values
    y = df["account"].values      # variable objetivo: tiene cuenta
    s = df["female"].values       # variable sensible: sexo

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.30, random_state=seed, stratify=y
    )
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
    parser.add_argument("--data_path", type=str, default="data/findex_bolivia_2021_processed.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df = df.fillna(0)
    os.makedirs("results", exist_ok=True)
    records = []

    for seed in range(args.n_seeds):
        (X_tr, y_tr, s_tr,
         X_val, y_val, s_val,
         X_te, y_te, s_te) = split_data(df, seed)

        for lam in args.lambda_grid:
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

    pd.DataFrame(records).to_csv("results/run_results_bolivia.csv", index=False)
    print("✅ Resultados guardados en results/run_results.csv")

if __name__ == "__main__":
    main()
