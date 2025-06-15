
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Parámetros fijos para evitar errores de argparse
n = 5000
imbalance = 0.3
dp_gap = 0.2
seed = 42

# Configuración validada manualmente
X, y = make_classification(n_samples=n,
                           n_features=10,
                           n_informative=2,
                           n_redundant=0,
                           n_repeated=0,
                           weights=[1 - imbalance, imbalance],
                           random_state=seed)

# Variable sensible correlacionada con clase
s = np.copy(y)
flip_mask = np.random.rand(n) < dp_gap
s[flip_mask] = 1 - s[flip_mask]

# Guardar CSV
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
df["y"] = y
df["s"] = s
df.to_csv("data/synthetic_data.csv", index=False)
print("✅ Archivo generado: data/synthetic_data.csv")
