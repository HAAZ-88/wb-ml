
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/run_results.csv")

# Prepara la carpeta
os.makedirs("results", exist_ok=True)

# Define el orden de lambda para asegurar consistencia
order = sorted(df["lambda_norm"].unique())

# Figura 1: Accuracy y Macro-F1
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="lambda_norm", y="Accuracy", ci=95, capsize=.2, label="Accuracy")
sns.barplot(data=df, x="lambda_norm", y="MacroF1", ci=95, capsize=.2, color="skyblue", label="Macro-F1")
plt.xlabel("$\lambda_{\text{norm}}$")
plt.ylabel("Score")
plt.title("Precisión del modelo")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_precision.png")
plt.clf()

# Figura 2: Brechas de equidad
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="lambda_norm", y="TPR_gap", ci=95, capsize=.2, label="TPR-gap")
sns.barplot(data=df, x="lambda_norm", y="DP_gap", ci=95, capsize=.2, color="orange", label="DP-gap")
plt.xlabel("$\lambda_{\text{norm}}$")
plt.ylabel("Gap")
plt.title("Equidad intergrupal")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_equity.png")
plt.clf()

# Figura 3: Bienestar social
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="lambda_norm", y="Welfare", ci=95, capsize=.2)
plt.xlabel("$\lambda_{\text{norm}}$")
plt.ylabel("Welfare $W$")
plt.title("Bienestar social (prioritarista)")
plt.tight_layout()
plt.savefig("results/fig_welfare.png")
plt.clf()

print("✅ Figuras exportadas a la carpeta results/")
