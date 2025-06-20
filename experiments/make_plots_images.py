
"""Genera gráficos PNG (no PDF) de las métricas clave para datos sintéticos y para Bolivia.

Uso:
    python make_plots_images.py
Los archivos se guardan en results/ como:
    plot_accuracy_<set>.png
    plot_macroF1_<set>.png
    plot_tprgap_<set>.png
    plot_dpgap_<set>.png
    plot_welfare_<set>.png
donde <set> es 'synthetic' o 'bolivia'.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metric(df, metric, ylabel, label, fname):
    mean = df.groupby("lambda_norm")[metric].mean()
    std  = df.groupby("lambda_norm")[metric].std()
    n    = df.groupby("lambda_norm")[metric].count()
    ci   = 1.96 * std / (n ** 0.5)

    x = mean.index.values
    y = mean.values
    yerr = ci.values

    plt.figure(figsize=(6,4))
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=4)
    plt.xlabel("lambda_norm")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs lambda_norm ({label})")
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"results/{fname}", dpi=300)
    plt.close()
    print(f"✅ saved results/{fname}")

def process(file_path, label):
    df = pd.read_csv(file_path)
    plot_metric(df, "Accuracy", "Accuracy", label, f"plot_accuracy_{label}.png")
    plot_metric(df, "MacroF1", "Macro-F1", label, f"plot_macroF1_{label}.png")
    plot_metric(df, "TPR_gap", "TPR-gap", label, f"plot_tprgap_{label}.png")
    plot_metric(df, "DP_gap", "DP-gap", label, f"plot_dpgap_{label}.png")
    plot_metric(df, "Welfare", "Welfare", label, f"plot_welfare_{label}.png")

if __name__ == "__main__":
    sns.set(style="whitegrid")
    process("results/run_results_synthetic.csv", "synthetic")
    process("results/run_results_bolivia.csv", "bolivia")
