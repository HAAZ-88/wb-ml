
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Parameters for reproducibility
n            = 7000
base_pos     = 0.30
dp_gap       = 0.15      # moderate disparity
flip_y       = 0.05      # label noise (5%)
n_features   = 6
n_informative= 2
seed         = 42
rng          = np.random.default_rng(seed)

# Sensitive attribute (0/1) 50-50
s = rng.integers(0, 2, size=n)

# Base features and labels
X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        weights=[1-base_pos, base_pos],
        random_state=seed)

# Impose disparity: group 1 gets extra positives
mask = (s == 1) & (rng.random(n) < dp_gap)
y[mask] = 1

# Add label noise
noise_mask = rng.random(n) < flip_y
y[noise_mask] = 1 - y[noise_mask]

# Save
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
df["y"] = y
df["s"] = s
df.to_csv("data/synthetic_data.csv", index=False)
print("✅ synthetic_data.csv generated (dp_gap=0.08, noise=0.05)")
