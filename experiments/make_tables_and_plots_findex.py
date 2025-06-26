import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

DOMINIOS = ["synthetic", "bolivia", "peru", "colombia", "ecuador"]
NOMBRES = {
    "synthetic": "Sintético",
    "bolivia": "Bolivia",
    "peru": "Perú",
    "colombia": "Colombia",
    "ecuador": "Ecuador"
}

RESULTS_DIR = "results"
OUT_TABLES_DIR = os.path.join(RESULTS_DIR, "tablas")
OUT_PLOTS_DIR = os.path.join(RESULTS_DIR, "figuras")
os.makedirs(OUT_TABLES_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

def format_row(mean, std, factor=1.0, precision=3):
    return f"{factor * mean:.{precision}f} ± {factor * std:.{precision}f}"

def resumir_tabla(df, nombre, agrupador):
    lambdas = sorted(df[agrupador].unique())
    tabla = []

    for lam in lambdas:
        subset = df[df[agrupador] == lam]
        fila = [lam]
        for col in ["Accuracy", "MacroF1", "TPR_gap", "DP_gap", "Welfare"]:
            mean = subset[col].mean()
            std = subset[col].std()
            factor = 1e8 if col == "Welfare" else 1.0
            prec = 2 if col == "Welfare" else 3
            fila.append(format_row(mean, std, factor, prec))
        tabla.append(fila)

    columnas = [
        r"$\lambda_{\mathrm{norm}}$",
        "Accuracy", "Macro-F1", "TPR-gap", "DP-gap", "Welfare $W^*$ ($\\times 10^{-8}$)"
    ]
    df_out = pd.DataFrame(tabla, columns=columnas)
    df_out.to_csv(f"{OUT_TABLES_DIR}/table_{nombre}.csv", index=False)

def graficar_metricas(df, nombre, agrupador):
    grouped = df.groupby(agrupador)
    lambdas = sorted(grouped.groups.keys())

    def mean_ci(metric):
        m = grouped[metric].mean()
        e = 1.96 * grouped[metric].std() / len(grouped) ** 0.5
        return m, e

    acc_m, acc_e = mean_ci("Accuracy")
    f1_m, f1_e = mean_ci("MacroF1")

    plt.figure()
    plt.errorbar(lambdas, acc_m, yerr=acc_e, label="Accuracy", fmt="o-", capsize=4)
    plt.errorbar(lambdas, f1_m, yerr=f1_e, label="Macro-F1", fmt="s--", capsize=4)
    plt.xlabel(r"$\lambda_{\mathrm{norm}}$")
    plt.ylabel("Precisión")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_PLOTS_DIR}/plot_{nombre}_accuracy.png")
    plt.close()

    tpr_m, tpr_e = mean_ci("TPR_gap")
    dp_m, dp_e = mean_ci("DP_gap")
    w_m, w_e = mean_ci("Welfare")

    plt.figure()
    plt.errorbar(lambdas, tpr_m, yerr=tpr_e, label="TPR-gap", fmt="o-", capsize=4)
    plt.errorbar(lambdas, dp_m, yerr=dp_e, label="DP-gap", fmt="s--", capsize=4)
    plt.errorbar(lambdas, w_m, yerr=w_e, label=r"Welfare $W^*$", fmt="^:", capsize=4)
    plt.xlabel(r"$\lambda_{\mathrm{norm}}$")
    plt.ylabel("Equidad y bienestar ($W^*$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_PLOTS_DIR}/plot_{nombre}_fairness_welfare.png")
    plt.close()

def tabla_comparativa(df, dominio):
    base = df[df["lambda_hat"] == 0]
    opt = df.copy()

    def delta(col):
        base_m = base[col].mean()
        opt_m = opt[col].mean()
        return 100 * (opt_m - base_m) / abs(base_m)

    return {
        "Dominio": NOMBRES[dominio],
        r"$\Delta$ Accuracy (%)": delta("Accuracy"),
        r"$\Delta$ Macro-F1 (%)": delta("MacroF1"),
        "Reducción TPR-gap (%)": -delta("TPR_gap"),
        "Reducción DP-gap (%)": -delta("DP_gap"),
        "Mejora Welfare $W^*$ (%)": delta("Welfare")
    }

comparativas = []

for dom in DOMINIOS:
    path_grid = f"{RESULTS_DIR}/run_results_{dom}_grid.csv"
    path_opt = f"{RESULTS_DIR}/run_results_{dom}.csv"

    if os.path.exists(path_grid):
        df = pd.read_csv(path_grid)
        agrupador = "lambda_norm"
    else:
        df = pd.read_csv(path_opt)
        agrupador = "lambda_hat"

    resumir_tabla(df, dom, agrupador)
    graficar_metricas(df, dom, agrupador)

    if os.path.exists(path_opt):
        df_opt = pd.read_csv(path_opt)
        comparativas.append(tabla_comparativa(df_opt, dom))

# Tabla resumen final
pd.DataFrame(comparativas).round(2).to_csv(f"{OUT_TABLES_DIR}/table_comparison_summary.csv", index=False)
print("✅ Tablas y gráficos generados para todos los dominios, incluyendo el sintético")