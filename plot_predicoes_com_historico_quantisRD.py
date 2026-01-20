# grafico_tft_previsoes.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# Caminhos dos arquivos
BASE_PATH = "C:/projects/tcc/"
HIST_PATH = os.path.join(BASE_PATH, "dados_top10_consolidadosRD_2015_2024.csv")
PRED_PATH = os.path.join(BASE_PATH, "predicoes_TFT_RD_v1.csv")
OUT_DIR = os.path.join(BASE_PATH, "graficos_predicoes_quantis_RD")

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------
# 1. Carregar os dados
# ----------------------------------------------------
print("📥 Carregando dados históricos e previsões...")
df_hist = pd.read_csv(HIST_PATH)
df_pred = pd.read_csv(PRED_PATH)

# ----------------------------------------------------
# 2. Garantir consistência de tipos
# ----------------------------------------------------
df_hist["PROC_REA"] = df_hist["PROC_REA"].astype(str)
df_pred["PROC_REA"] = df_pred["PROC_REA"].astype(str)

df_hist["Ano"] = df_hist["Ano"].astype(int)
df_hist["Mes"] = df_hist["Mes"].astype(int)
df_pred["Ano"] = df_pred["Ano"].astype(int)
df_pred["Mes"] = df_pred["Mes"].astype(int)

df_hist = df_hist.groupby(["PROC_REA", "PROC_DESCR", "Ano", "Mes"]).agg(
    Valor_Total=("Valor_Total", "sum")
).reset_index()

# ----------------------------------------------------
# 3. Criar eixo temporal contínuo (AAAAMM)
# ----------------------------------------------------
df_hist["Data"] = pd.to_datetime(df_hist["Ano"].astype(str) + "-" + df_hist["Mes"].astype(str) + "-01")
df_pred["Data"] = pd.to_datetime(df_pred["Ano"].astype(str) + "-" + df_pred["Mes"].astype(str) + "-01")

# ----------------------------------------------------
# 4. Ordenar e consolidar
# ----------------------------------------------------
df_hist = df_hist.sort_values(["PROC_REA", "Data"])
df_pred = df_pred.sort_values(["PROC_REA", "Data"])
# filtrar previsões para 2025-2026
#df_pred = df_pred[(df_pred["Ano"] >= 2025) & (df_pred["Ano"] <= 2026)]

df_pred["Valor_Previsto_Mediana"] = pd.to_numeric(df_pred["Valor_Previsto_Mediana"], errors="coerce").astype("float32")
df_pred["Valor_Melhor_Cenario"] = pd.to_numeric(df_pred["Valor_Melhor_Cenario"], errors="coerce").astype("float32")
df_pred["Valor_Pior_Cenario"] = pd.to_numeric(df_pred["Valor_Pior_Cenario"], errors="coerce").astype("float32")
#df_pred["Valor_Total"] *= 10

#proc_id = "303010223"
proc_id = "303010037"
filtro = (df_hist["PROC_REA"] == proc_id) & (df_hist["Ano"] == 2024) & (df_hist["Mes"] == 1)
valor = df_hist.loc[filtro, "Valor_Total"]
print(f"Valor histórico para o procedimento {proc_id} em Jan/2024: R$ {valor.values[0]}")
filtro = (df_hist["PROC_REA"] == proc_id) & (df_hist["Ano"] == 2024) & (df_hist["Mes"] == 2)
valor = df_hist.loc[filtro, "Valor_Total"]
print(f"Valor histórico para o procedimento {proc_id} em Fev/2024: R$ {valor.values[0]}")

filtro = (df_pred["PROC_REA"] == proc_id) & (df_pred["Ano"] == 2025) & (df_pred["Mes"] == 1)
valor = df_pred.loc[filtro, "Valor_Previsto_Mediana"]
print(f"Valor previsto para o procedimento {proc_id} em Jan/2025: R$ {valor.values[0]}")
filtro = (df_pred["PROC_REA"] == proc_id) & (df_pred["Ano"] == 2025) & (df_pred["Mes"] == 2)
valor = df_pred.loc[filtro, "Valor_Previsto_Mediana"]
print(f"Valor previsto para o procedimento {proc_id} em Fev/2025: R$ {valor.values[0]}")
# ----------------------------------------------------
# 5. Gerar gráficos por procedimento
# ----------------------------------------------------
procedures = sorted(df_hist["PROC_REA"].unique())
print(f"📊 Gerando gráficos para {len(procedures)} procedimentos...")

for proc in procedures:
    hist = df_hist[df_hist["PROC_REA"] == proc]
    pred = df_pred[df_pred["PROC_REA"] == proc]

    if hist.empty or pred.empty:
        print(f"⚠️ Pulando {proc} — sem dados históricos ou previsões.")
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(hist["Data"], hist["Valor_Total"], label="Histórico (2015–2024)", color="blue", linewidth=2)
    plt.plot(pred["Data"], pred["Valor_Previsto_Mediana"], label="Previsão (2025–2034)", color="orange", linestyle="--", linewidth=2)
    plt.plot(pred["Data"], pred["Valor_Melhor_Cenario"], label="Melhor Cenário", color="green", linestyle="--", linewidth=1)
    plt.plot(pred["Data"], pred["Valor_Pior_Cenario"], label="Pior Cenário", color="red", linestyle="--", linewidth=1)

    plt.title(f"Procedimento {proc}: {hist['PROC_DESCR'].iloc[0]} — Valor Total (R$)")
    plt.xlabel("Ano")
    plt.ylabel("Valor Total (R$)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # salva gráfico individual
    fname = os.path.join(OUT_DIR, f"grafico_{proc}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

print(f"✅ Gráficos salvos em: {OUT_DIR}")
