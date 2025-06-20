
import pandas as pd
import os

def format_ci(series):
    mean = series.mean()
    ci = 1.96 * series.std() / (len(series) ** 0.5)
    return f"{mean:.3f} ± {ci:.3f}"

def process_table(file_path, label):
    df = pd.read_csv(file_path)
    grouped = df.groupby("lambda_norm")
    table = pd.DataFrame({
        "Accuracy": grouped["Accuracy"].apply(format_ci),
        "Macro-F1": grouped["MacroF1"].apply(format_ci),
        "TPR-gap": grouped["TPR_gap"].apply(format_ci),
        "DP-gap": grouped["DP_gap"].apply(format_ci),
        "Welfare": grouped["Welfare"].apply(format_ci)
    })
    table.index.name = "λ"
    out_path = f"results/table_results_{label}.csv"
    table.to_csv(out_path)
    print(f"✅ Tabla guardada en {out_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    process_table("results/run_results_synthetic.csv", "synthetic")
    process_table("results/run_results_bolivia.csv", "bolivia")
