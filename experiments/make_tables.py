
import pandas as pd
import numpy as np

def format_ci(series):
    mean = series.mean()
    std = series.std(ddof=1)
    ci = 1.96 * std / np.sqrt(len(series))
    return f"${mean:.3f} \, \pm \, {ci:.3f}$"

def format_scientific(series):
    mean = series.mean()
    std = series.std(ddof=1)
    ci = 1.96 * std / np.sqrt(len(series))
    return f"$ {mean:.2e} \, \pm \, {ci:.0f} $"

if __name__ == "__main__":
    df = pd.read_csv("results/run_results.csv")
    grouped = df.groupby("lambda_norm")

    table = pd.DataFrame({
        "Accuracy": grouped["accuracy"].apply(format_ci),
        "Macro-F1": grouped["macro_f1"].apply(format_ci),
        "TPR-gap": grouped["tpr_gap"].apply(format_ci),
        "DP-gap": grouped["dp_gap"].apply(format_ci),
        "Bienestar": grouped["welfare"].apply(format_scientific)
    })

    table.index.name = "$\lambda_{\text{norm}}$"

    latex = table.to_latex(escape=False, column_format="lccccc")
    with open("results/table_results.tex", "w") as f:
        f.write(latex)

    print("✅ Tabla exportada a LaTeX en: results/table_results.tex")
