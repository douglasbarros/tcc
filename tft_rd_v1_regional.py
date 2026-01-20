# ==========================================================
# TFT v1_regional — otimizado para GPU 6GB
# Previsão de Jan/2025 até Dez/2034 (120 meses)
# Inclui:
#   ✅ AGREGAÇÃO MENSAL DOS DADOS
#   ✅ NÍVEL 3: PREVISÃO COM QUANTIS
#   ✅ Integração de features externas
#   ✅ MUDANÇA REGIONAL: Granularidade por Unidade da Federação (Região)
# ==========================================================

import multiprocessing
import os, gc, time, traceback
import numpy as np
import pandas as pd
import psutil
import torch

from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping

# Imports para avaliação e benchmarking
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================================
# NOVO: Função para carregar e processar dados reais do IBGE
# ==========================================================
def carregar_e_processar_projecao_ibge(caminho_csv):
    """
    Lê o arquivo de projeção do IBGE, filtra, transforma de wide para long,
    agrega por faixa etária e sexo, e pivota para criar features.
    """
    print("IBGE: Carregando e processando projeções populacionais...")
    try:
        # Tenta ler com encoding padrão, se falhar, tenta 'latin-1' comum no Brasil
        try:
            df_ibge = pd.read_csv(caminho_csv, sep=';')
        except UnicodeDecodeError:
            df_ibge = pd.read_csv(caminho_csv, sep=';', encoding='latin-1')

        # 1. Filtragem inicial
        df_ibge = df_ibge[
            (df_ibge['LOCAL'] == 'Brasil') &
            (df_ibge['SEXO'].isin(['Homens', 'Mulheres']))
        ].copy()

        # Garantir que a coluna 'IDADE' é numérica, ignorando 'Total' ou outras strings
        df_ibge = df_ibge[pd.to_numeric(df_ibge['IDADE'], errors='coerce').notna()]
        df_ibge['IDADE'] = df_ibge['IDADE'].astype(int)

        # 2. Transformar de "wide" para "long"
        anos_cols = [str(ano) for ano in range(2000, 2071)]
        df_long = df_ibge.melt(
            id_vars=['IDADE', 'SEXO'],
            value_vars=anos_cols,
            var_name='Ano',
            value_name='Populacao'
        )

        # 3. Limpeza dos dados
        df_long['Ano'] = df_long['Ano'].astype(int)
        # Limpa espaços em branco e converte para número
        df_long['Populacao'] = df_long['Populacao'].astype(str).str.replace(' ', '').str.replace('.', '', regex=False).astype(int)

        # 4. Criar faixas etárias
        bins = [-1, 19, 59, 150] # Bins para 0-19, 20-59, 60+
        labels = ['0_19', '20_59', '60_mais']
        df_long['faixa_etaria'] = pd.cut(df_long['IDADE'], bins=bins, labels=labels, right=True)

        # 5. Agregar por Ano, faixa etária e sexo
        df_agg = df_long.groupby(['Ano', 'faixa_etaria', 'SEXO'])['Populacao'].sum().reset_index()

        # 6. Pivotar para criar as colunas de features
        df_agg['SEXO'] = df_agg['SEXO'].str.lower()
        df_agg['feature_name'] = 'pop_' + df_agg['faixa_etaria'].astype(str) + '_' + df_agg['SEXO']
        
        df_pivot = df_agg.pivot_table(index='Ano', columns='feature_name', values='Populacao').reset_index()
        df_pivot.columns.name = None # Limpa o nome do índice das colunas
        
        print("IBGE: Processamento concluído com sucesso.")
        return df_pivot

    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo de projeção do IBGE não encontrado em '{caminho_csv}'")
        return None
    except Exception as e:
        print(f"❌ ERRO ao processar arquivo do IBGE: {e}")
        return None

def gerar_projecoes_economicas(anos):
    # Esta função pega dados de IPCA e PIB do boletim focus - Relatório de Mercado - 07/11/2025
    # https://www.bcb.gov.br/publicacoes/focus
    dados_base = {
        2015: {'ipca': 10.67, 'pib': -3.5}, 2016: {'ipca': 6.29, 'pib': -3.3},
        2017: {'ipca': 2.95, 'pib': 1.3}, 2018: {'ipca': 3.75, 'pib': 1.8},
        2019: {'ipca': 4.31, 'pib': 1.2}, 2020: {'ipca': 4.52, 'pib': -3.3},
        2021: {'ipca': 10.06, 'pib': 5.0}, 2022: {'ipca': 5.79, 'pib': 3.0},
        2023: {'ipca': 4.5, 'pib': 2.9}, 2024: {'ipca': 3.8, 'pib': 2.0},
        2025: {'ipca': 4.55, 'pib': 2.16}, 2026: {'ipca': 4.2, 'pib': 1.78},
        2027: {'ipca': 3.8, 'pib': 1.88}, 2028: {'ipca': 3.5, 'pib': 2.0},
    }
    longo_prazo = {'ipca': 3.0, 'pib': 2.0}
    dados = []
    for ano in anos:
        if ano in dados_base:
            dados.append({'Ano': ano, 'ipca_anual': dados_base[ano]['ipca'], 'pib_anual': dados_base[ano]['pib']})
        else:
            dados.append({'Ano': ano, 'ipca_anual': longo_prazo['ipca'], 'pib_anual': longo_prazo['pib']})
    return pd.DataFrame(dados)

# ==========================================================
# Função para carregar e processar dados epidemiológicos
# ==========================================================
def carregar_e_processar_dados_epidemiologicos(caminho_base, anos):
    """
    Carrega e processa múltiplos arquivos de dados epidemiológicos do DATASUS.
    """
    print(" EPI: Carregando e processando dados epidemiológicos...")

    # Iniciar com um calendário completo de Ano e Mês.
    df_final_epi = pd.DataFrame([(ano, mes) for ano in anos for mes in range(1, 13)], columns=['Ano', 'Mes'])

    # Dicionário de arquivos e nomes de features
    arquivos_epi = {
        'dengue_brasil_2015_2024.csv': 'casos_dengue',
        'zika_brasil_2016_2025.csv': 'casos_zika',
        'chikungunya_brasil_2017_2025.csv': 'casos_chikungunya',
        'esquistossomose_brasil_2015_2024.csv': 'casos_esquistossomose',
        'hepatite_brasil_2015_2023.csv': 'casos_hepatite',
        'meningite_brasil_2015_2023.csv': 'casos_meningite',
        'tuberculose_brasil_2015_2024.csv': 'casos_tuberculose',
    }

    for arquivo, nome_feature in arquivos_epi.items():
        try:
            caminho_arquivo = os.path.join(caminho_base, arquivo)
            # Os CSVs do TabNet podem ter cabeçalhos e rodapés extras. `skiprows` e `skipfooter` podem ser necessários.
            # Também podem usar ';' como separador e encoding 'latin-1'.
            df_doenca = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1', engine='python')
            
            nome_original_ano = df_doenca.columns[0]
            df_doenca.rename(columns={nome_original_ano: 'Ano'}, inplace=True)
            # Transformar de wide para long
            df_doenca = df_doenca.melt(id_vars=['Ano'], var_name='Mes', value_name=nome_feature)
            # remove coluna Total
            df_doenca = df_doenca[df_doenca['Mes'] != 'Total']
            
            # Limpeza
            df_doenca = df_doenca[pd.to_numeric(df_doenca['Ano'], errors='coerce').notna()]
            df_doenca['Ano'] = df_doenca['Ano'].astype(int)
            
            # Mapear nome do mês para número
            mapa_meses = {'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12}
            df_doenca['Mes'] = df_doenca['Mes'].map(mapa_meses)
            df_doenca = df_doenca.dropna(subset=['Mes'])
            df_doenca['Mes'] = df_doenca['Mes'].astype(int)
            
            # Lidar com valores não numéricos (ex: '-') e converter para int
            df_doenca[nome_feature] = pd.to_numeric(df_doenca[nome_feature], errors='coerce').fillna(0).astype(int)

            # Juntar com o DataFrame final
            df_final_epi = pd.merge(df_final_epi, df_doenca, on=['Ano', 'Mes'], how='left')

        except FileNotFoundError:
            print(f" EPI: Arquivo '{arquivo}' não encontrado. Pulando.")
        except Exception as e:
            print(f" EPI: Erro ao processar '{arquivo}': {e}")
    
    # Preenche meses futuros sem dados com 0
    df_final_epi = df_final_epi.fillna(0)
    print(" EPI: Processamento concluído.")
    return df_final_epi

# ==========================================================
# NOVO: Função para carregar e processar dados de COVID-19
# ==========================================================
def carregar_e_processar_dados_covid(caminho_base, anos_historicos):
    """
    Lê múltiplos arquivos anuais de COVID, limpa, agrega UFs,
    converte de semana para mês e retorna um DataFrame mensal.
    """
    print(" COVID: Carregando e processando dados de COVID-19...")
    all_covid_data = []

    for ano in anos_historicos:
        try:
            caminho_arquivo = os.path.join(caminho_base, f"covid{ano}.CSV")
            df_ano = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1')

            # 1. Padronizar nomes de colunas
            df_ano.rename(columns={
                'Casos novos notificados na semana epidemiológica': 'casos_novos_semana',
                'Óbitos novos notificados na semana epidemiológica': 'obitos_novos_semana'
            }, inplace=True)

            # Manter apenas as colunas que importam
            if 'casos_novos_semana' not in df_ano.columns:
                print(f" COVID: Coluna de casos novos não encontrada em covid{ano}.CSV. Pulando.")
                continue
                
            df_ano = df_ano[['casos_novos_semana']].copy()

            # 2. Limpar números (formato brasileiro para numérico)
            for col in df_ano.columns:
                df_ano[col] = df_ano[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                df_ano[col] = pd.to_numeric(df_ano[col], errors='coerce').fillna(0)

            # 3. Adicionar a semana epidemiológica (assumindo que as linhas estão em ordem)
            df_ano['semana_epi'] = range(1, len(df_ano) + 1)
            df_ano['Ano'] = ano

            # 4. Converter semana para data e extrair o mês
            # Cria uma data para o início de cada semana epidemiológica
            df_ano['data_semana'] = pd.to_datetime(df_ano['Ano'].astype(str) + '-' + df_ano['semana_epi'].astype(str) + '-1', format='%Y-%U-%w')
            df_ano['Mes'] = df_ano['data_semana'].dt.month
            
            # 5. Agregar para o nível nacional (somando UFs) e mensal
            df_mensal = df_ano.groupby(['Ano', 'Mes'])['casos_novos_semana'].sum().reset_index()
            df_mensal.rename(columns={'casos_novos_semana': 'casos_covid_mes'}, inplace=True)
            
            all_covid_data.append(df_mensal)

        except FileNotFoundError:
            print(f" COVID: Arquivo 'covid{ano}.CSV' não encontrado. Pulando.")
        except Exception as e:
            print(f" COVID: Erro ao processar 'covid{ano}.CSV': {e}")

    if not all_covid_data:
        print(" COVID: Nenhum dado de COVID foi carregado.")
        return pd.DataFrame(columns=['Ano', 'Mes', 'casos_covid_mes'])

    df_covid_final = pd.concat(all_covid_data, ignore_index=True)
    print(" COVID: Processamento concluído.")
    return df_covid_final

if __name__ == "__main__":
    multiprocessing.freeze_support()
    BASE_PATH = "C:/projects/tcc/"
    CSV_PATH = BASE_PATH + "dados_top10_consolidadosRD_2015_2024.csv"
    PATH_POP_IBGE = BASE_PATH + "projecoes_2024_tab1_idade_simples.csv" # Caminho para o arquivo do IBGE
    PATH_DOENCAS = BASE_PATH + "doencas/"
    OUT_PATH = BASE_PATH + "predicoes_TFT_RD_v1_regional.csv"
    CHECKPOINT_PATH = BASE_PATH + "tft_RD_v1_regional.ckpt"
    
    seed_everything(42)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("medium")
    
    max_encoder_length = 48
    min_encoder_length = max_encoder_length // 2
    max_prediction_length = 12
    validation_months = 24
    hidden_size = 32
    hidden_continuous_size = 16
    attention_head_size = 4
    lstm_layers = 1
    batch_size = 128
    max_epochs = 100 # Aumente se necessário para o modelo aprender os quantis
    num_workers = 4
    persistent_workers = True
    pin_memory = True
    
    # Definir o callback de Parada Antecipada
    early_stop_callback = EarlyStopping(
        monitor="val_loss",   # Métrica a ser observada
        min_delta=1e-4,       # A melhora mínima para ser considerada uma melhora
        patience=5,           # Número de épocas para esperar por uma melhora antes de parar
        verbose=True,         # Imprimir mensagens quando parar
        mode="min"            # O modo é 'min' porque queremos minimizar a perda
    )

    de_para_uf = {
        '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
        '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE', '29': 'BA',
        '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
        '41': 'PR', '42': 'SC', '43': 'RS',
        '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'
    }

    de_para_regiao = {
        '1': 'Norte', '2': 'Nordeste', '3': 'Sudeste', '4': 'Sul', '5': 'Centro-Oeste'
    }
    
    # ----------------------------------------------------------
    # 5) Carregar, AGREGAR e ENRIQUECER os dados
    # ----------------------------------------------------------
    print("📥 Carregando e agregando dados de gastos...")
    # MUDANÇA REGIONAL: Carregar a coluna 'Regiao' do CSV original
    usecols = ["PROC_REA", "Regiao", "Ano", "Mes", "Valor_Total"]
    df_raw = pd.read_csv(CSV_PATH, low_memory=False, usecols=usecols)
    
    df_raw["PROC_REA"] = df_raw["PROC_REA"].astype(str)
    df_raw['Regiao'] = df_raw['Regiao'].astype(str)
    df_raw['UF'] = df_raw['Regiao'].map(de_para_uf)
    df_raw['Nome_Regiao'] = df_raw['Regiao'].str[0].map(de_para_regiao)
    df_raw["Valor_Total"] = pd.to_numeric(df_raw["Valor_Total"], errors="coerce").astype("float32")
    df_raw = df_raw.dropna(subset=["Valor_Total", "Ano", "Mes", "PROC_REA", "Nome_Regiao"])

    print("🔄 Agregando dados por (Procedimento, Nome_Regiao, Ano, Mês)...")
    # MUDANÇA REGIONAL: Adicionar 'Nome_Regiao' ao groupby
    df = df_raw.groupby(["PROC_REA", "Nome_Regiao", "Ano", "Mes"]).agg(
        Valor_Total=("Valor_Total", "sum")
    ).reset_index()
    del df_raw; gc.collect()

    # ==========================================================
    # Integração de todos os dados externos
    # Criar e juntar os dados de projeção do IBGE, Econômicos e Epidemiológicos
    # ==========================================================
    print("📈 Gerando e juntando projeções futuras...")
    anos_totais = range(2015, 2035)
    df_pop = carregar_e_processar_projecao_ibge(PATH_POP_IBGE)
    df_econ = gerar_projecoes_economicas(anos_totais)
    
    if df_pop is not None:
        df_projecoes = pd.merge(df_pop, df_econ, on="Ano")
        df = pd.merge(df, df_projecoes, on="Ano", how="left")
        print("✅ Dados enriquecidos com projeções do IBGE e econômicas.")
    else:
        print("⚠️ Não foi possível carregar os dados do IBGE. Prosseguindo sem eles.")
        df_projecoes = df_econ # Usar apenas dados econômicos como fallback
        df = pd.merge(df, df_projecoes, on="Ano", how="left")
    
    # Carregar dados epidemiológicos
    print("📈 Carregando e juntando dados epidemiológicos...")
    df_epi = carregar_e_processar_dados_epidemiologicos(PATH_DOENCAS, anos_totais)
    df = pd.merge(df, df_epi, on=["Ano", "Mes"], how="left")

    df_covid = carregar_e_processar_dados_covid(PATH_DOENCAS, range(2020, 2026))
    df = pd.merge(df, df_covid, on=["Ano", "Mes"], how="left")

    df = df.sort_values(["PROC_REA", "Nome_Regiao", "Ano", "Mes"]).reset_index(drop=True)
    # MUDANÇA REGIONAL: O time_idx agora é calculado para cada grupo (procedimento, Nome_Regiao)
    df["time_idx"] = df.groupby(["PROC_REA", "Nome_Regiao"]).cumcount().astype("int32")

    print(f"Dados agregados para {len(df.groupby(['PROC_REA', 'Nome_Regiao']))} grupos (séries temporais).")
    print(f"time_idx máximo (meses): {df['time_idx'].max()}")

    # ----------------------------------------------------------
    # 6) Construir Datasets de Treino e Validação
    # ----------------------------------------------------------
    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - validation_months

    # MUDANÇA REGIONAL: Adicionar 'Nome_Regiao' como uma feature estática categórica
    static_features = ["PROC_REA", "Nome_Regiao"]
    
    # ==========================================================
    # Lista de colunas para o modelo, agora com features do IBGE
    # ==========================================================
    known_future_reals = [
        "time_idx", "Ano", "Mes",
        # Features do IBGE (se carregadas com sucesso)
        'pop_0_19_homens', 'pop_0_19_mulheres',
        'pop_20_59_homens', 'pop_20_59_mulheres',
        'pop_60_mais_homens', 'pop_60_mais_mulheres',
        # Features Econômicas
        "ipca_anual", "pib_anual",
        
    ]
    # Filtra a lista para incluir apenas as colunas que realmente existem no DataFrame
    known_future_reals = [col for col in known_future_reals if col in df.columns]
    unknown_future_reals = [
        "Valor_Total",
        # Features Epidemiológicas
        "casos_dengue", "casos_zika", "casos_chikungunya",
        "casos_esquistossomose", "casos_hepatite", "casos_meningite", "casos_tuberculose"
    ]

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Valor_Total",
        # MUDANÇA REGIONAL: Definir os novos grupos
        group_ids=["PROC_REA", "Nome_Regiao"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        # known_reals e unknown_reals permanecem os mesmos
        time_varying_known_reals=known_future_reals,
        time_varying_unknown_reals=unknown_future_reals,
        # MUDANÇA REGIONAL: Adicionar 'Nome_Regiao' às categorias estáticas
        static_categoricals=static_features,
        allow_missing_timesteps=True
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    # ----------------------------------------------------------
    # 7) Criar ou carregar o modelo
    # ----------------------------------------------------------

    model_exists = os.path.exists(CHECKPOINT_PATH)
    if model_exists:
        print("✅ Modelo encontrado! Carregando:", CHECKPOINT_PATH)
        tft = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    else:
        print("⚙️ Criando modelo TFT regional...")
        quantiles = [0.1, 0.5, 0.9]
        tft = TemporalFusionTransformer.from_dataset(
            training, hidden_size=hidden_size, hidden_continuous_size=hidden_continuous_size,
            attention_head_size=attention_head_size, lstm_layers=lstm_layers, dropout=0.1,
            loss=QuantileLoss(quantiles=quantiles), output_size=len(quantiles),
            learning_rate=3e-4, reduce_on_plateau_patience=3
        )
        trainer = Trainer(accelerator="auto", devices="auto", precision=32, max_epochs=max_epochs, gradient_clip_val=0.1, callbacks=[early_stop_callback])
        print("🚀 Iniciando treino regional…")
        trainer.fit(tft, train_loader, val_loader)
        print("💾 Salvando modelo regional…")
        trainer.save_checkpoint(CHECKPOINT_PATH)
    
    tft.to("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------
    # 8) AVALIAÇÃO E BENCHMARKING POR REGIÃO (NOVA SEÇÃO)
    # ----------------------------------------------------------
    print("\n📊 Iniciando Avaliação e Benchmarking Regional no Conjunto de Validação...")

    df_train = df[lambda x: x.time_idx <= training_cutoff].copy()
    df_val = df[lambda x: x.time_idx > training_cutoff].copy()

    # --- 8.1) Fazer previsões do TFT e RECONSTRUIR o índice ---
    print("  - Gerando previsões do TFT e reconstruindo o índice para o conjunto de validação...")
    try:
        # PASSO 1: Obter todas as previsões
        with torch.inference_mode():
            # A saída é apenas o dicionário de previsões
            raw_predictions_dict = tft.predict(val_loader, mode="raw")
            predictions_list = raw_predictions_dict["prediction"]
        
        # PASSO 2: Reconstruir o DataFrame de índice manualmente
        index_dfs = []
        # Iteramos pelo val_loader para extrair o índice de cada batch
        for x, y in iter(val_loader):
            # A biblioteca fornece uma função auxiliar para converter a entrada 'x' em um índice legível
            index_dfs.append(validation.x_to_index(x))
        
        index_df = pd.concat(index_dfs, ignore_index=True)

        # Isolar a previsão da mediana
        tft_preds_median = predictions_list[:, :, 1].flatten().cpu().numpy()
        
        # Obter os valores reais alinhados do dataloader
        actuals_flat = torch.cat([y[0] for x, y in iter(val_loader)]).flatten().cpu().numpy()

        # Criar um DataFrame com os resultados para facilitar a filtragem
        results_tft = index_df.loc[index_df.index.repeat(max_prediction_length)].reset_index(drop=True)
        
        # Garantir que todos os arrays tenham o mesmo comprimento
        min_len = min(len(results_tft), len(actuals_flat), len(tft_preds_median))
        results_tft = results_tft.iloc[:min_len]
        results_tft['actuals'] = actuals_flat[:min_len]
        results_tft['preds_tft'] = tft_preds_median[:min_len]

    except Exception as e:
        print(f"  ❌ Erro fatal ao gerar previsões do TFT. A avaliação não pode continuar. Erro: {e}")
        traceback.print_exc()
        exit()

    # --- 8.2) Loop de Avaliação por Região ---
    evaluation_results = []
    coluna_geografica = "Nome_Regiao"
    regioes = sorted(df_val[coluna_geografica].unique())

    for regiao in regioes:
        print(f"\n  - Avaliando Região: {regiao}...")
        
        # Filtrar dados e resultados para a região atual
        df_train_region = df_train[df_train[coluna_geografica] == regiao]
        df_val_region = df_val[df_val[coluna_geografica] == regiao]
        results_tft_region = results_tft[results_tft[coluna_geografica] == regiao]

        if df_val_region.empty:
            print(f"    -> Sem dados de validação para a região {regiao}. Pulando.")
            continue
            
        # Calcular métricas do TFT para a região
        mae_tft = mean_absolute_error(results_tft_region['actuals'], results_tft_region['preds_tft'])
        rmse_tft = np.sqrt(mean_squared_error(results_tft_region['actuals'], results_tft_region['preds_tft']))
        mape_tft = mean_absolute_percentage_error(results_tft_region['actuals'] + 1e-6, results_tft_region['preds_tft'])
        
        # Treinar e avaliar o Baseline de Regressão Linear para a região
        features = ['time_idx', 'Mes', 'PROC_REA']
        target = 'Valor_Total'
        X_train, y_train = df_train_region[features], df_train_region[target]
        X_val, y_val = df_val_region[features], df_val_region[target]
        
        preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Mes', 'PROC_REA'])], remainder='passthrough')
        lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
        
        lr_pipeline.fit(X_train, y_train)
        lr_preds = lr_pipeline.predict(X_val)
        
        mae_lr = mean_absolute_error(y_val, lr_preds)
        rmse_lr = np.sqrt(mean_squared_error(y_val, lr_preds))
        mape_lr = mean_absolute_percentage_error(y_val + 1e-6, lr_preds)
        
        # Armazenar resultados
        evaluation_results.append({
            'Região': regiao, 'MAE_TFT': mae_tft, 'MAE_LR': mae_lr,
            'RMSE_TFT': rmse_tft, 'RMSE_LR': rmse_lr,
            'MAPE_TFT': mape_tft, 'MAPE_LR': mape_lr
        })

    # --- 8.3) Apresentar Resultados Regionais ---
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results)
        
        print("\n" + "="*80)
        print("Resultados da Avaliação Regional (no período de Jan/2023 a Dez/2024)")
        print("="*80)
        
        # Tabela de MAE
        print("\n--- Erro Absoluto Médio (MAE) por Região (em R$) ---")
        print(f"{'Região':<15} | {'TFT (Seu Modelo)':<20} | {'Regressão Linear':<20} | {'Melhoria (%)':<15}")
        print("-"*80)
        for _, row in results_df.iterrows():
            melhoria = (row['MAE_LR'] - row['MAE_TFT']) / row['MAE_LR'] if row['MAE_LR'] > 0 else 0
            print(f"{row['Região']:<15} | R$ {row['MAE_TFT']:>15,.2f} | R$ {row['MAE_LR']:>15,.2f} | {melhoria:>13.2%}")

        # Tabela de MAPE
        print("\n--- Erro Percentual Absoluto Médio (MAPE) por Região ---")
        print(f"{'Região':<15} | {'TFT (Seu Modelo)':<20} | {'Regressão Linear':<20} | {'Melhoria (p.p.)':<15}")
        print("-"*80)
        for _, row in results_df.iterrows():
            melhoria_pp = row['MAPE_LR'] - row['MAPE_TFT']
            print(f"{row['Região']:<15} | {row['MAPE_TFT']:>19.2%} | {row['MAPE_LR']:>19.2%} | {melhoria_pp:>13.2%}")
        
        print("="*80 + "\n")
    else:
        print("  - Nenhuma avaliação regional foi gerada.")

    # ==========================================================
    # NOVO: Seção 8.4 - Análise do Horizonte de Previsão
    # ==========================================================
    print("\n" + "="*75)
    print("Análise de Performance por Horizonte de Previsão (Backtesting)")
    print("="*75)
    print("Simulando uma previsão de 24 meses a partir do final de 2022...")

    # DataFrame de partida é o histórico de treino
    df_backtest_hist = df_train.copy()
    # Lista para armazenar as previsões do backtest
    backtest_predictions = []

    # Loop por cada grupo para gerar a previsão recursiva de 24 meses
    coluna_geografica = "Nome_Regiao"
    grupos_backtest = df_backtest_hist.groupby(["PROC_REA", coluna_geografica])

    for i, (group_ids, df_g) in enumerate(grupos_backtest, start=1):
        proc_id, geo_id = group_ids
        print(f"  - Backtesting para ({proc_id}, {geo_id})...")

        # Pular grupos com histórico muito curto
        if len(df_g) < min_encoder_length:
            print(f"    -> Histórico curto ({len(df_g)} meses). Pulando.")
            continue
        
        # Lógica de previsão recursiva, idêntica à da inferência final
        df_local = df_g.copy()
        produced = 0
        while produced < validation_months: # Prever por 24 meses
            step_horizon = min(max_prediction_length, validation_months - produced)
            try:
                with torch.inference_mode():
                    new_predictions = tft.predict(df_local, mode="raw")["prediction"]
                
                preds_block = new_predictions[0][:step_horizon].cpu().numpy().astype("float32")
                valor_mediana = preds_block[:, 1]
            except Exception:
                # Se a previsão falhar para um step, interrompe para este grupo
                break

            # Construir o bloco futuro
            last_time_idx = df_local["time_idx"].max()
            last_ano = df_local["Ano"].iloc[-1]
            last_mes = df_local["Mes"].iloc[-1]

            future_time_idx = range(last_time_idx + 1, last_time_idx + 1 + step_horizon)
            future_years, future_months = [], []
            temp_ano, temp_mes = last_ano, last_mes
            for _ in range(step_horizon):
                temp_mes += 1
                if temp_mes > 12: temp_mes = 1; temp_ano += 1
                future_years.append(temp_ano)
                future_months.append(temp_mes)

            out_block = pd.DataFrame({
                "PROC_REA": proc_id, coluna_geografica: geo_id,
                "time_idx": list(future_time_idx), "Ano": future_years, "Mes": future_months,
                "Valor_Previsto_Mediana": valor_mediana
            })
            
            # Juntar features externas e placeholders para a próxima iteração
            out_block_for_concat = pd.merge(out_block, df_projecoes, on="Ano", how="left")
            out_block_for_concat["Valor_Total"] = out_block_for_concat["Valor_Previsto_Mediana"]
            for col in unknown_future_reals:
                if col not in out_block_for_concat.columns:
                    out_block_for_concat[col] = 0.0

            df_local = pd.concat([df_local, out_block_for_concat], ignore_index=True)
            produced += step_horizon
        
        # Armazenar apenas a parte prevista
        backtest_predictions.append(df_local[df_local['time_idx'] > df_g['time_idx'].max()])

    # Unir todas as previsões do backtest
    if backtest_predictions:
        df_backtest_results = pd.concat(backtest_predictions, ignore_index=True)

        # Juntar previsões com os valores reais do conjunto de validação
        df_val_comparison = pd.merge(
            df_val[["PROC_REA", coluna_geografica, "Ano", "Mes", "Valor_Total"]],
            df_backtest_results[["PROC_REA", coluna_geografica, "Ano", "Mes", "Valor_Previsto_Mediana"]],
            on=["PROC_REA", coluna_geografica, "Ano", "Mes"]
        )

        # Separar por ano
        val_ano_1 = df_val_comparison[df_val_comparison['Ano'] == 2023]
        val_ano_2 = df_val_comparison[df_val_comparison['Ano'] == 2024]

        # Calcular métricas para o Ano 1
        mae_ano_1 = mean_absolute_error(val_ano_1['Valor_Total'], val_ano_1['Valor_Previsto_Mediana'])
        mape_ano_1 = mean_absolute_percentage_error(val_ano_1['Valor_Total'] + 1e-6, val_ano_1['Valor_Previsto_Mediana'])

        # Calcular métricas para o Ano 2
        mae_ano_2 = mean_absolute_error(val_ano_2['Valor_Total'], val_ano_2['Valor_Previsto_Mediana'])
        mape_ano_2 = mean_absolute_percentage_error(val_ano_2['Valor_Total'] + 1e-6, val_ano_2['Valor_Previsto_Mediana'])

        # Apresentar resultados
        print("\n--- Performance do TFT por Ano de Previsão ---")
        print(f"{'Período':<15} | {'MAE (Erro Médio em R$)':<25} | {'MAPE (Erro Médio %)'}")
        print("-"*75)
        print(f"{'Ano 1 (2023)':<15} | R$ {mae_ano_1:>22,.2f} | {mape_ano_1:>15.2%}")
        print(f"{'Ano 2 (2024)':<15} | R$ {mae_ano_2:>22,.2f} | {mape_ano_2:>15.2%}")
        print("-"*75)

        aumento_mape = (mape_ano_2 - mape_ano_1)
        print(f"Conclusão: O erro percentual (MAPE) aumentou em {aumento_mape:.2%} do primeiro para o segundo ano de previsão.")
        print("Isso demonstra o decaimento esperado da acurácia conforme o horizonte de previsão se estende.")
        print("="*75 + "\n")
    else:
        print("  - Nenhuma previsão de backtesting foi gerada.")

    # ----------------------------------------------------------
    # 9) Inferência — prever 120 meses (2025–2034)
    # ----------------------------------------------------------
    print("\n🔮 Prevendo recursivamente 2025–2034 por (Procedimento, Nome_Regiao)...")
    all_outputs = []
    
    # MUDANÇA REGIONAL: Iterar sobre os novos grupos
    grupos = df.groupby(["PROC_REA", "Nome_Regiao"])

    for i, (group_ids, df_g) in enumerate(grupos, start=1):
        proc_id, nome_regiao = group_ids
        print(f"[{i}/{len(grupos)}] → Prev {proc_id} em {nome_regiao}")
        
        if df_g.empty: continue

        previsao_sucedida = True
        
        months_needed = 120
        produced = 0
        df_local = df_g.copy()
        # filtra os dados para o grupo atual
        df_local = df_local[(df_local["PROC_REA"] == proc_id) & (df_local["Nome_Regiao"] == nome_regiao)].reset_index(drop=True)
        df_local["time_idx"] = df_local.groupby(["PROC_REA", "Nome_Regiao"]).cumcount().astype("int32")
        last_real_time_idx = df_local["time_idx"].max()
        
        while produced < months_needed:
            step_horizon = min(max_prediction_length, months_needed - produced)
            try:
                with torch.inference_mode():
                    new_predictions = tft.predict(df_local, mode="raw")["prediction"]
                
                preds_block = new_predictions[0][:step_horizon].cpu().numpy().astype("float32")
                valor_melhor_cenario, valor_mediana, valor_pior_cenario = preds_block[:, 0], preds_block[:, 1], preds_block[:, 2]
            except (AssertionError, IndexError) as e:
                if "filters should not remove" in str(e) or "index out of range" in str(e):
                    print(f"  ⚠️ AVISO: Dados insuficientes ou esparsos para o grupo ({proc_id}, {nome_regiao}).")
                    print("     A série histórica pode ser muito curta ou ter muitos meses faltando.")
                    print("     Pulando a previsão para este grupo.")
                    previsao_sucedida = False # Marca a previsão como falha
                    break # Sai do loop 'while' para este grupo
                else: # Se for outro tipo de erro, ainda queremos saber
                    print(f"  ❌ Erro inesperado durante tft.predict() para o grupo ({proc_id}, {nome_regiao}): {e}"); traceback.print_exc()
                    previsao_sucedida = False
                    break
            except Exception as e:
                print(f"  ❌ Erro genérico durante tft.predict() para o grupo ({proc_id}, {nome_regiao}): {e}"); traceback.print_exc()
                previsao_sucedida = False
                break
            
            current_future_time_idx_start = df_local["time_idx"].max() + 1
            future_time_idx = range(current_future_time_idx_start, current_future_time_idx_start + step_horizon)
            temp_month, temp_year = int(df_local["Mes"].iloc[-1]), int(df_local["Ano"].iloc[-1])
            future_years, future_months = [], []
            for _ in range(step_horizon):
                temp_month += 1
                if temp_month > 12: temp_month = 1; temp_year += 1
                future_years.append(temp_year)
                future_months.append(temp_month)
            
            out_block = pd.DataFrame({
                "PROC_REA": [proc_id] * len(valor_mediana),
                "time_idx": list(future_time_idx)[:len(valor_mediana)],
                "Nome_Regiao": [nome_regiao] * len(valor_mediana),
                "Ano": future_years[:len(valor_mediana)], "Mes": future_months[:len(valor_mediana)],
                "Valor_Previsto_Mediana": valor_mediana, "Valor_Melhor_Cenario": valor_melhor_cenario,
                "Valor_Pior_Cenario": valor_pior_cenario,
            })

            # Atualizar o bloco futuro com placeholders para as novas features
            out_block_for_concat = out_block.copy()
            out_block_for_concat = pd.merge(out_block_for_concat, df_projecoes, on="Ano", how="left")
        
            out_block_for_concat["Valor_Total"] = out_block_for_concat["Valor_Previsto_Mediana"]
            # Adicionar placeholder para as features desconhecidas. 0 é uma escolha neutra.
            # Como não sabemos os futuros casos das doencas, usamos 0.0. O modelo aprende que 0.0 no futuro é um valor neutro e que ele deve confiar nos outros padrões que aprendeu para fazer a previsão.
            out_block_for_concat["casos_dengue"] = 0.0
            out_block_for_concat["casos_zika"] = 0.0
            out_block_for_concat["casos_chikungunya"] = 0.0
            out_block_for_concat["casos_esquistossomose"] = 0.0
            out_block_for_concat["casos_hepatite"] = 0.0
            out_block_for_concat["casos_meningite"] = 0.0
            out_block_for_concat["casos_tuberculose"] = 0.0
            
            df_local = pd.concat([df_local, out_block_for_concat], ignore_index=True)
            produced += len(valor_mediana)
            if produced < months_needed: print(f"   produzido {produced}/{months_needed} meses (até {out_block['Ano'].iloc[-1]}-{out_block['Mes'].iloc[-1]:02d})")
            del new_predictions, preds_block, out_block, out_block_for_concat
            gc.collect()
        
        if previsao_sucedida:
            df_future_result = df_local[df_local["time_idx"] > last_real_time_idx].iloc[:months_needed].reset_index(drop=True)
            final_cols = ["PROC_REA", "Nome_Regiao", "time_idx", "Ano", "Mes", "Valor_Previsto_Mediana", "Valor_Melhor_Cenario", "Valor_Pior_Cenario"]
            
            # Adicionar verificação se as colunas existem antes de tentar acessá-las
            if all(col in df_future_result.columns for col in final_cols):
                all_outputs.append(df_future_result[final_cols])
            else:
                print(f"  ⚠️ AVISO: As colunas de previsão não foram geradas corretamente para o grupo ({proc_id}, {nome_regiao}). Não adicionando ao resultado final.")
        
        del df_local, df_g
        gc.collect()
        
    if all_outputs:
        final = pd.concat(all_outputs, ignore_index=True)
        final.to_csv(OUT_PATH, index=False)
        print(f"\n✅ Previsões regionais salvas em:", OUT_PATH)
    else:
        print("❌ Nenhuma previsão gerada.")
    print("✅ Fim.")