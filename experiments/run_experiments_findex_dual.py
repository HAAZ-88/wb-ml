import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import WBLogisticModel
from src.metrics import tpr_gap, dp_gap

def welfare_index(y_true, y_pred, gamma=1.0, rho=2):
    u = np.where(y_true == y_pred, gamma, 1e-6)
    return np.sum(u ** (1 - rho) / (1 - rho))

def split_data(df, seed):
    cols = ["educ", "age", "inc_q", "emp_in", "mobileowner", "internetaccess"]
    X = df[cols].values
    y = df["account"].values
    s = df["female"].values
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    return X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test

def evaluate(model, X, y, s, gamma, rho):
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "MacroF1": f1_score(y, y_pred, average="macro"),
        "TPR_gap": tpr_gap(y, y_pred, s),
        "DP_gap": dp_gap(y_pred, s),
        "Welfare": welfare_index(y, y_pred, gamma, rho)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--lambda_grid", nargs="+", type=float, required=True)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path).fillna(0)
    os.makedirs("results", exist_ok=True)

    country = os.path.basename(args.data_path).split("_")[1]
    out_all = f"results/run_results_{country}_grid.csv"
    out_opt = f"results/run_results_{country}.csv"

    # Calcular máximo Welfare para normalización
    df_temp = []
    for lam in args.lambda_grid:
        model = WBLogisticModel(lambda_norm=lam, C=1.0)
        model.fit(*split_data(df, 0)[:3])
        val_metrics = evaluate(model, *split_data(df, 0)[3:6], args.gamma, args.rho)
        df_temp.append(val_metrics['Welfare'])
    max_welfare = max(df_temp) if df_temp else 1.0

    all_rows = []
    final_rows = []

    for seed in range(args.n_seeds):
        X_tr, y_tr, s_tr, X_val, y_val, s_val, X_te, y_te, s_te = split_data(df, seed)

        models = {}
        val_scores = {}

        for lam in args.lambda_grid:
            model = WBLogisticModel(lambda_norm=lam, C=1.0)
            model.fit(X_tr, y_tr, s_tr)
            val_metrics = evaluate(model, X_val, y_val, s_val, args.gamma, args.rho)
            norm_welfare = val_metrics["Welfare"] / max_welfare
            score = val_metrics["Accuracy"] - 2 * val_metrics["TPR_gap"] + 0.2 * norm_welfare
            val_scores[lam] = score
            models[lam] = model

            test_metrics = evaluate(model, X_te, y_te, s_te, args.gamma, args.rho)
            test_metrics.update({"seed": seed, "lambda_norm": lam})
            all_rows.append(test_metrics)

        best_lam = max(val_scores, key=val_scores.get)
        best_model = models[best_lam]
        best_metrics = evaluate(best_model, X_te, y_te, s_te, args.gamma, args.rho)
        best_metrics.update({"seed": seed, "lambda_hat": best_lam})
        final_rows.append(best_metrics)

    pd.DataFrame(all_rows).to_csv(out_all, index=False)
    pd.DataFrame(final_rows).to_csv(out_opt, index=False)
    print(f"✅ Resultados completos guardados en {out_all}")
    print(f"✅ Resultados óptimos por semilla guardados en {out_opt}")

if __name__ == "__main__":
    main()