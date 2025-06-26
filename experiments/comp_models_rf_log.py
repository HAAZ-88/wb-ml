
import pandas as pd
import os

# Archivos esperados
archivos = {
    "Bolivia_log": "results/run_results_bolivia.csv",
    "Bolivia_rf": "results/run_results_bolivia_rf.csv",
    "Ecuador_log": "results/run_results_ecuador.csv",
    "Ecuador_rf": "results/run_results_ecuador_rf.csv"
}

resultados = {}

for nombre, path in archivos.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        medias = df[["Accuracy", "MacroF1", "TPR_gap", "DP_gap", "Welfare"]].mean()
        resultados[nombre] = medias
    else:
        print(f"⚠️ Archivo no encontrado: {path}")

# Construir tabla comparativa
df_final = pd.DataFrame(resultados).T.round(4)
df_final = df_final[["Accuracy", "MacroF1", "TPR_gap", "DP_gap", "Welfare"]]
df_final.to_csv("results/tabla_comparativa_log_rf.csv")
print("✅ Tabla comparativa guardada en: results/tabla_comparativa_log_rf.csv")
