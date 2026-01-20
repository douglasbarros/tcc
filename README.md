# 📊 Análise e Projeção dos Gastos do SUS com Temporal Fusion Transformer (TFT)

Este repositório contém os códigos-fonte, scripts de processamento e visualização utilizados no Trabalho de Conclusão de Curso (TCC):

**“Análise e Projeção da Curva de Gastos do SUS com Base em Procedimentos Ambulatoriais e de Internação Hospitalar (2015–2024)”**

O estudo utiliza modelos de séries temporais avançados, com destaque para o **Temporal Fusion Transformer (TFT)**, visando gerar previsões probabilísticas dos gastos do Sistema Único de Saúde (SUS), em nível **nacional e regional**, como subsídio ao planejamento orçamentário e à gestão de risco em políticas públicas de saúde.

---

## 🎯 Objetivos do Projeto

- Modelar a evolução histórica dos gastos do SUS com procedimentos:
  - **Ambulatoriais (PA)**
  - **Internações Hospitalares (RD)**
- Comparar o desempenho do **Temporal Fusion Transformer (TFT)** com modelos de baseline (Regressão Linear)
- Gerar **previsões probabilísticas** utilizando quantis (0.1, 0.5 e 0.9)
- Avaliar diferenças entre:
  - Modelagem **nacional**
  - Modelagem **regional** (Centro-Oeste, Nordeste, Norte, Sudeste e Sul)
- Apoiar análises de:
  - Sazonalidade
  - Incerteza
  - Planejamento orçamentário
  - Gestão de risco (cenários otimista, mediano e pessimista)

---

## 🧠 Metodologia

- **Modelo principal:** Temporal Fusion Transformer (TFT)
- **Função de perda:** Quantile Loss
- **Horizonte de previsão:** até 10 anos
- **Validação:** Backtesting (2023–2024)
- **Métricas de avaliação:**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

Os dados históricos foram organizados por procedimento e enriquecidos com variáveis exógenas, permitindo capturar padrões complexos de longo e curto prazo.

---

## 📁 Estrutura do Repositório

### 🔹 Scripts de Treinamento dos Modelos

- **Modelo nacional — Procedimentos Ambulatoriais (PA)**  
  `tft_pa_v1.py`

- **Modelo nacional — Internações Hospitalares (RD)**  
  `tft_rd_v1.py`

- **Modelo regional — Procedimentos Ambulatoriais (PA)**  
  `tft_pa_v1_regional.py`

- **Modelo regional — Internações Hospitalares (RD)**  
  `tft_rd_v1_regional.py`

---

### 🔹 Scripts de Visualização e Análise

- **Gráficos de previsões — PA (Nacional)**  
  `plot_predicoes_com_historico_quantis.py`

- **Gráficos de previsões — RD (Nacional)**  
  `plot_predicoes_com_historico_quantisRD.py`

- **Gráficos de previsões — PA (Regional)**  
  `plot_predicoes_com_historico_quantis_regional.py`

- **Gráficos de previsões — RD (Regional)**  
  `plot_predicoes_com_historico_quantis_regionalRD.py`

Os gráficos apresentam:
- Série histórica (2015–2024)
- Previsões futuras
- Cenários probabilísticos (quantis 0.1, 0.5 e 0.9)

---

## 📈 Resultados Principais

- O modelo **TFT superou amplamente a regressão linear**, reduzindo o MAPE em mais de 50% em diversos cenários
- A modelagem **regional apresentou maior robustez** em horizontes de previsão mais longos
- As previsões probabilísticas permitem:
  - Planejamento baseado na mediana
  - Cálculo de reservas de contingência
  - Identificação de procedimentos e regiões com maior volatilidade
- Procedimentos como **parto normal e cesariana** apresentaram projeções crescentes, mesmo após períodos históricos de queda, acompanhadas de maior incerteza (distância entre quantis)

---

## ⚠️ Observações Importantes

- Procedimentos fortemente impactados por choques exógenos, como o **Tratamento de Infecção pelo Coronavírus (COVID-19)**, não foram utilizados para projeções de longo prazo devido à curta janela histórica (2020–2022).
- O foco do estudo é **previsão**, não inferência causal.

---

## 🔁 Reprodutibilidade

Todos os scripts utilizados na modelagem e visualização estão disponíveis neste repositório, garantindo:

- Transparência metodológica
- Reprodutibilidade dos experimentos
- Possibilidade de extensão futura do estudo

---

## 🛠️ Tecnologias Utilizadas

- Python
- PyTorch / PyTorch Forecasting
- Temporal Fusion Transformer (TFT)
- Pandas, NumPy
- Matplotlib
- Scikit-learn

---

## 📌 Autor

**Douglas de Barros Silva**  
Desenvolvedor de Software | Tech Lead  
Trabalho de Conclusão de Curso — Análise de Dados / Ciência de Dados

---

## 📄 Licença

Este projeto é disponibilizado para fins acadêmicos e educacionais.
