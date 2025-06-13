
# experiments/run_experiments.py
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score

from src.model import WBLogisticModel
from src.metrics import tpr_gap, dp_gap

def welfare_index(y_true, y_pred, gamma=1.0, rho=2):
    u = np.where(y_true == y_pred, gamma, 0.0)
    return np.sum(u ** (1 - rho) / (1 - rho))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lambda_grid', nargs='+', type=float, required=True)
parser.add_argument('--n_seeds', type=int, default=10)
args = parser.parse_args()

# Load data
df = pd.read_csv('data/synthetic_data.csv')
X = df[[col for col in df.columns if col.startswith('x')]].values
y = df['y'].values
s = df['s'].values

# Create results folder if needed
os.makedirs('results', exist_ok=True)

# Main loop
records = []
for lambda_norm in args.lambda_grid:
    for seed in range(args.n_seeds):
        np.random.seed(seed)

        # Instanciar modelo
        model = WBLogisticModel(lambda_norm=lambda_norm)

        # Entrenar
        model.fit(X, y, s)

        # Predecir
        y_pred = model.predict(X)

        # Métricas
        acc = accuracy_score(y, y_pred)
        macrof1 = f1_score(y, y_pred, average='macro')
        tprgap = tpr_gap(y, y_pred, s)
        dpgap = dp_gap(y_pred, s)
        welfare = welfare_index(y, y_pred)

        # Registrar
        records.append({
            'lambda_norm': lambda_norm,
            'seed': seed,
            'Accuracy': acc,
            'MacroF1': macrof1,
            'TPR_gap': tprgap,
            'DP_gap': dpgap,
            'Welfare': welfare
        })

# Guardar CSV
df_results = pd.DataFrame(records)
df_results.to_csv('results/run_results.csv', index=False)
print('Resultados guardados en results/run_results.csv')
