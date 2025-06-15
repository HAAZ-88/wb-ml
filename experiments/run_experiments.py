
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

def simulate_fair_training(X, y, s, lambda_norm):
    # Dummy classifier: logistic regression + pseudo fairness loss
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    y_pred = model.predict(X)

    # Simulate fairness metrics and welfare
    tpr_gap = abs(y_pred[s == 0].mean() - y_pred[s == 1].mean())
    dp_gap = abs((y_pred[s == 0] == 1).mean() - (y_pred[s == 1] == 1).mean())
    welfare = -np.sum((y - y_pred) ** 2) - lambda_norm * (tpr_gap + dp_gap)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "tpr_gap": tpr_gap,
        "dp_gap": dp_gap,
        "welfare": welfare
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_grid", nargs="+", type=float, default=[0.0, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--n_seeds", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv("data/synthetic_data.csv")
    X = df[[col for col in df.columns if col.startswith("x")]].values
    y = df["y"].values
    s = df["s"].values

    os.makedirs("results", exist_ok=True)
    rows = []

    for lmbda in args.lambda_grid:
        for seed in range(args.n_seeds):
            np.random.seed(seed)
            metrics = simulate_fair_training(X, y, s, lambda_norm=lmbda)
            rows.append({
                "lambda_norm": lmbda,
                "seed": seed,
                **metrics
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv("results/run_results.csv", index=False)
    print("✅ Resultados guardados en results/run_results.csv")
