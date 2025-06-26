import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # sigmoid

# Parámetros del generador
n = 7000
n_features = 6
informative_idx = [0, 1, 2]   # Variables relevantes
label_noise = 0.05
fn_bias_prob = 0.15  # Prob. de forzar falso negativo si s=1
seed = 42

rng = np.random.default_rng(seed)

# === Generar X ===
X_raw = rng.normal(0, 1, size=(n, n_features))
X = StandardScaler().fit_transform(X_raw)

# === Variable sensible s: depende no linealmente de x0 y x1 ===
p_s = expit(1.5 * X[:, 0] - 1.2 * X[:, 1])  # probabilidad dependiente de x
s = rng.binomial(1, p_s)

# === Etiqueta base y (antes de sesgo) ===
logits = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.2 * X[:, 2] + 0.5 * s
p_y = expit(logits)
y_clean = rng.binomial(1, p_y)

# === Introducir sesgo estructural: falsos negativos para s=1 ===
# Si y_clean == 1 y s==1, hay una probabilidad de convertirlo en 0 (sesgo)
mask_fn = (y_clean == 1) & (s == 1) & (rng.random(n) < fn_bias_prob)
y_biased = y_clean.copy()
y_biased[mask_fn] = 0

# === Ruido aleatorio (flip label con probabilidad p) ===
flip_mask = rng.random(n) < label_noise
y_final = y_biased.copy()
y_final[flip_mask] = 1 - y_final[flip_mask]

# === Guardar como DataFrame ===
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
df["y"] = y_final
df["s"] = s

# === Guardar CSV ===
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_data.csv", index=False)
print("✅ synthetic_data_structured.csv generado correctamente")