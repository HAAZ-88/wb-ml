import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--imbalance', type=float, default=0.3)
parser.add_argument('--dp_gap', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
s = rng.integers(0, 2, size=args.n)
X, y = make_classification(n_samples=args.n,
                           n_features=5,
                           n_informative=3,
                           n_redundant=0,
                           weights=[1-args.imbalance, args.imbalance],
                           random_state=args.seed)
mask = (s==1) & (rng.random(args.n) < args.dp_gap)
y[mask] = 1 - y[mask]

df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
df['y'] = y
df['s'] = s
out = 'synthetic_data.csv'
df.to_csv(out, index=False)
print('Synthetic data saved to', out)
