
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--imbalance", type=float, default=0.3)
    parser.add_argument("--dp_gap", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Generar datos sintéticos asegurando suma < n_features
    X, y = make_classification(n_samples=args.n,
                               n_features=5,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               weights=[1 - args.imbalance, args.imbalance],
                               random_state=args.seed)

    # Crear variable sensible (grupo) correlacionada con clase
    s = np.copy(y)
    flip_mask = np.random.rand(args.n) < args.dp_gap
    s[flip_mask] = 1 - s[flip_mask]  # invertir clase para inducir gap

    # Guardar como CSV
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y
    df["s"] = s
    df.to_csv("data/synthetic_data.csv", index=False)
    print("Archivo generado: data/synthetic_data.csv")
