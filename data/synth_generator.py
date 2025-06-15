
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

    # Configuración segura: 10 > 2 + 0 + 0
    n_features = 10
    n_informative = 2
    n_redundant = 0
    n_repeated = 0

    X, y = make_classification(n_samples=args.n,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               n_repeated=n_repeated,
                               weights=[1 - args.imbalance, args.imbalance],
                               random_state=args.seed)

    assert X.shape[1] == n_features, "Verifica que las columnas generadas son correctas"

    # Crear variable sensible correlacionada artificialmente con y
    s = np.copy(y)
    flip_mask = np.random.rand(args.n) < args.dp_gap
    s[flip_mask] = 1 - s[flip_mask]

    # Guardar CSV
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["y"] = y
    df["s"] = s
    df.to_csv("data/synthetic_data.csv", index=False)
    print("✅ Archivo generado: data/synthetic_data.csv")
