
import pandas as pd
import numpy as np
import os

def format_ci(series):
    mean = series.mean()
    ci = 1.96 * series.std() / np.sqrt(len(series))
    return f"$ {mean:.3f} \, \pm \, {ci:.3f} $"

def format_ci_sci(series):
    mean = series.mean()
    ci = 1.96 * series.std() / np.sqrt(len(series))
    return f"$ {mean:.2e} \, \pm \, {ci:.0f} $"

df = pd.read_csv("results/run_results.csv")
grouped = df.groupby("lambda_norm")

table = pd.DataFrame({
    "Accuracy": grouped["Accuracy"].apply(format_ci),
    "Macro-F1": grouped["MacroF1"].apply(format_ci),
    "TPR-gap": grouped["TPR_gap"].apply(format_ci),
    "DP-gap": grouped["DP_gap"].apply(format_ci),
    "Welfare $W$": grouped["Welfare"].apply(format_ci_sci)
})

table.index.name = "$\lambda_{\text{norm}}$"
os.makedirs("results", exist_ok=True)
table.to_latex("results/table_results.tex", escape=False, column_format="lccccc")
print("✅ Tabla exportada a results/table_results.tex")
