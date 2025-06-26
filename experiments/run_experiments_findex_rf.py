
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import tpr_gap, dp_gap

def welfare_index(y_true, y_pred, gamma=1.0, rho=2):
    u = np.where(y_true == y_pred, gamma, 1e-6)
    return np.sum(u ** (1 - rho) / (1 - rho))

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--n_seeds", type=int, default=10)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--rho", type=float, default=2)
args = parser.parse_args()

df = pd.read_csv(args.data_path).fillna(0)
X = df[["educ", "age", "inc_q", "emp_in", "mobileowner", "internetaccess"]].values
y = df["account"].values
s = df["female"].values

results = []

for seed in range(args.n_seeds):
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=seed, stratify=y_temp)

    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "MacroF1": f1_score(y_test, y_pred, average="macro"),
        "TPR_gap": tpr_gap(y_test, y_pred, s_test),
        "DP_gap": dp_gap(y_pred, s_test),
        "Welfare": welfare_index(y_test, y_pred, gamma=args.gamma, rho=args.rho),
        "seed": seed
    }
    results.append(metrics)

df_out = pd.DataFrame(results)
country = os.path.basename(args.data_path).split("_")[1]
out_path = f"results/run_results_{country}_rf.csv"
df_out.to_csv(out_path, index=False)
print(f"✅ Resultados guardados en {out_path}")