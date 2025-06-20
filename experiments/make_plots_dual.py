
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metrics(df, label):
    metrics = ["Accuracy", "TPR_gap", "Welfare"]
    df_mean = df.groupby("lambda_norm")[metrics].mean().reset_index()
    df_std = df.groupby("lambda_norm")[metrics].std().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metrics:
        ax.errorbar(df_mean["lambda_norm"], df_mean[metric], 
                    yerr=1.96 * df_std[metric] / (df_std[metric].count() ** 0.5), 
                    label=metric, capsize=4, marker='o')

    ax.set_xlabel("λ (lambda_norm)")
    ax.set_ylabel("Score")
    ax.set_title(f"Desempeño del modelo WB-ML ({label})")
    ax.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/plot_performance_{label}.pdf")
    print(f"✅ Figura guardada en results/plot_performance_{label}.pdf")

if __name__ == "__main__":
    sns.set(style="whitegrid")
    df_synth = pd.read_csv("results/run_results_synthetic.csv")
    df_bol = pd.read_csv("results/run_results_bolivia.csv")
    plot_metrics(df_synth, "synthetic")
    plot_metrics(df_bol, "bolivia")
