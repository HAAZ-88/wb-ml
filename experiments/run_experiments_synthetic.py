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
    X = df[[c for c in df.columns if c.startswith("x")]].values
    y = df["y"].values
    s = df["s"].values
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    return (X_train, y_train, s_train,
            X_val, y_val, s_val,
            X_test, y_test, s_test)

def evaluate_model(model, X, y, s, gamma, rho):
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
    parser.add_argument("--lambda_grid", nargs="+", type=float, default=[0, 0.5, 1, 2, 5, 10])
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="data/synthetic_data.csv")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    os.makedirs("results", exist_ok=True)

    all_records = []   # solo modelos óptimos
    grid_records = []  # todos los modelos

    # Calcular máximo Welfare para normalización
    df_temp = []
    for lam in args.lambda_grid:
        model = WBLogisticModel(lambda_norm=lam, C=0.2)
        model.fit(*split_data(df, 0)[:3])
        val_metrics = evaluate_model(model, *split_data(df, 0)[3:6], args.gamma, args.rho)
        df_temp.append(val_metrics['Welfare'])
    max_welfare = max(df_temp) if df_temp else 1.0

    for seed in range(args.n_seeds):
        (X_tr, y_tr, s_tr,
         X_val, y_val, s_val,
         X_te, y_te, s_te) = split_data(df, seed)

        val_scores = {}
        models = {}

        for lam in args.lambda_grid:
            model = WBLogisticModel(lambda_norm=lam, C=0.2)
            model.fit(X_tr, y_tr, s_tr)

            # Evaluar en validación
            val_metrics = evaluate_model(model, X_val, y_val, s_val, args.gamma, args.rho)
            norm_welfare = val_metrics["Welfare"] / max_welfare
            score = val_metrics["Accuracy"] - 2 * val_metrics["TPR_gap"] + 0.2 * norm_welfare
            val_scores[lam] = score
            models[lam] = model

            # Evaluar en test y guardar en grid
            test_metrics = evaluate_model(model, X_te, y_te, s_te, args.gamma, args.rho)
            test_metrics.update({"seed": seed, "lambda_norm": lam})
            grid_records.append(test_metrics)

        best_lambda = max(val_scores, key=val_scores.get)
        best_model = models[best_lambda]
        best_metrics = evaluate_model(best_model, X_te, y_te, s_te, args.gamma, args.rho)
        best_metrics.update({"seed": seed, "lambda_hat": best_lambda})
        all_records.append(best_metrics)

    # Guardar archivos
    pd.DataFrame(grid_records).to_csv("results/run_results_synthetic_grid.csv", index=False)
    pd.DataFrame(all_records).to_csv("results/run_results_synthetic.csv", index=False)
    print("✅ Guardado: run_results_synthetic_grid.csv y run_results_synthetic.csv")
if __name__ == "__main__":
    main()