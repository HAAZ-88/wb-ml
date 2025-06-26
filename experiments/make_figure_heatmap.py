import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración visual
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

# Crear carpetas si no existen
os.makedirs("results/figuras", exist_ok=True)

# Cargar tabla comparativa
df = pd.read_csv("results/tablas/table_comparison_summary.csv")

# Renombrar columnas para el gráfico
df = df.rename(columns={
    "Dominio": "Dominio",
    r"$\Delta$ Accuracy (%)": "Accuracy",
    r"$\Delta$ Macro-F1 (%)": "Macro-F1",
    "Reducción TPR-gap (%)": "TPR-gap ↓",
    "Reducción DP-gap (%)": "DP-gap ↓",
    "Mejora Welfare $W^*$ (%)": "Welfare $W^*$"
})

df = df.set_index("Dominio")

# Crear heatmap
plt.figure(figsize=(10, 4.5))
sns.heatmap(df, annot=True, cmap="RdYlGn", center=0, fmt=".1f",
            linewidths=0.5, cbar_kws={"label": "% mejora o reducción"})
plt.ylabel("")
plt.tight_layout()

# Guardar figura
output_path = "results/figuras/heatmap_resultados.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Figura guardada en: {output_path}")