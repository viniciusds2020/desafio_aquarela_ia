# Resumo dos Processos do Repositório — Desafio Aquarela IA

## Visão Geral

Este repositório implementa uma solução completa de Machine Learning para **previsão de consumo energético residencial**, abrangendo desde a ingestão de dados brutos até a geração de predições e dashboards interativos.

- **Período dos dados:** Janeiro a Junho de 2023 (180 dias)
- **Clientes:** 100 consumidores residenciais
- **Regiões:** 5 regiões geográficas
- **Modelos treinados:** Random Forest e LightGBM

---

## 1. Ingestão de Dados (`ingest`)

Carrega três arquivos CSV:

| Arquivo | Registros | Colunas | Descrição |
|---------|-----------|---------|-----------|
| `consumo.csv` | 18.000 | 3 | Consumo diário por cliente (kWh) |
| `clima.csv` | 900 | 4 | Temperatura e umidade por região/dia |
| `clientes.csv` | 100 | 2 | Mapeamento cliente → região |

---

## 2. Transformação e Feature Engineering (`transform`)

### 2.1 Tratamento de Qualidade

- **Regiões desconhecidas:** 5 clientes com região "Desconhecida" foram reatribuídos à região mais similar com base no padrão de consumo.
- **Temperaturas faltantes:** 45 valores (~5%) interpolados linearmente dentro de cada região (ffill/bfill).

### 2.2 Criação de Features (SQL via DuckDB)

20 features engenheiradas usando funções de janela SQL:

| Categoria | Features | Descrição |
|-----------|----------|-----------|
| **Temporais** | `month`, `day_of_week`, `day_of_month`, `week_of_year`, `is_weekend` | Componentes de calendário |
| **Lags autoregressivos** | `consumption_lag1`, `lag2`, `lag3`, `lag7` | Consumo em dias anteriores |
| **Médias móveis** | `consumption_ma7`, `ma14`, `ma30` | Tendências de curto/médio prazo |
| **Estatísticas móveis** | `consumption_std7`, `min7`, `max7` | Variabilidade recente |
| **Climáticas** | `temperature`, `humidity`, `temp_lag1` | Condições meteorológicas |
| **Categórica** | `region_encoded` | Região codificada (Label Encoding) |

---

## 3. Treinamento de Modelos (`train`)

### 3.1 Estratégia de Validação

- **Divisão temporal:** Treino (Jan–Mai 2023) / Teste (Junho 2023)
- **Validação cruzada:** Janela expansível mês a mês para evitar vazamento temporal
- **~13.000 amostras** de treino, **~2.000** de teste

### 3.2 Modelos Comparados

| Modelo | Hiperparâmetros Principais |
|--------|---------------------------|
| **Random Forest** | 200 árvores, max_depth=15, min_samples_split=5 |
| **LightGBM** | 300 iterações, learning_rate=0.05, max_depth=10 |

### 3.3 Resultados no Conjunto de Teste

| Métrica | Random Forest | LightGBM | Vencedor |
|---------|---------------|----------|----------|
| **MAE (kWh)** | 1.86 | 1.86 | Empate |
| **RMSE (kWh)** | 2.33 | 2.33 | LightGBM (marginal) |
| **R²** | 0.6117 | 0.6134 | LightGBM |
| **MAPE (%)** | 13.91% | 13.81% | LightGBM |

**Modelo selecionado:** LightGBM (ligeiramente superior em RMSE e R²).

---

## 4. Inferência (`inference`)

- Carrega o melhor modelo (`best_model.pkl`)
- Gera predições em lote sobre os dados processados
- Salva resultados em `data/predictions.parquet`

---

## 5. Análise Exploratória (Notebook `01_eda.ipynb`)

- Distribuição de consumo e estatísticas descritivas
- Análise de valores ausentes e inconsistências
- Padrões temporais (sazonalidade, dia da semana)
- Variação regional de consumo
- Correlação clima × consumo
- Perfilamento de clientes

---

## 6. Dashboard Interativo (`dashboards/dashboard.ipynb`)

- Comparação de métricas entre modelos
- Predições vs. valores reais (série temporal e scatter)
- Análise de erro por região
- Heatmap de consumo regional
- Explorador interativo por cliente
- Correlação temperatura × consumo

---

## 7. Geração de Apresentação (`src/generate_presentation.py`)

Gera automaticamente `apresentacao.pdf` com 10 slides cobrindo contexto, metodologia, resultados, insights e próximos passos.

---

## Stack Tecnológica

| Categoria | Tecnologias |
|-----------|-------------|
| Processamento | Pandas, NumPy, DuckDB |
| ML | Scikit-learn, LightGBM, XGBoost |
| MLOps | MLflow (rastreamento de experimentos) |
| Visualização | Plotly, Matplotlib, Seaborn |
| Relatórios | fpdf2 |
| Notebooks | Jupyter |

---

## Fluxo de Execução

```
consumo.csv ─┐
clima.csv   ─┼─→ ingest() → transform() → train() → inference()
clientes.csv ┘        │          │           │           │
                       ▼          ▼           ▼           ▼
                  DataFrames   DuckDB     MLflow     predictions
                              features   modelos     .parquet
```

---

# Perguntas Simuladas sobre os Resultados

## P1: Qual modelo apresentou melhor desempenho e por quê?

**R:** O LightGBM foi selecionado como melhor modelo, embora a diferença seja marginal. Ele apresentou R² de 0.6134 contra 0.6117 do Random Forest, e MAPE de 13.81% contra 13.91%. A vantagem do LightGBM está na sua eficiência computacional (modelo de ~895 KB vs ~52 MB do RF) e capacidade de capturar interações não-lineares de forma mais eficiente com gradient boosting.

---

## P2: O R² de ~0.61 é satisfatório? Como poderia ser melhorado?

**R:** Um R² de 0.61 indica que o modelo explica cerca de 61% da variabilidade do consumo, o que é razoável para previsão energética residencial com dados limitados a 6 meses. Possíveis melhorias incluem:

- **Mais dados históricos** (pelo menos 1-2 anos para capturar sazonalidade completa)
- **Features externas** (feriados, preços de energia, dados socioeconômicos)
- **Modelos de séries temporais** (Prophet, LSTM) que capturam tendências de longo prazo
- **Ensemble de modelos** combinando diferentes abordagens
- **Granularidade horária** se disponível nos dados de consumo

---

## P3: Como o pipeline lida com o vazamento temporal (data leakage)?

**R:** O pipeline adota duas estratégias para evitar data leakage:

1. **Divisão temporal estrita:** O treino usa dados de Jan–Mai e o teste usa exclusivamente Junho, garantindo que nenhuma informação futura influencie o treinamento.
2. **Validação com janela expansível:** Durante o cross-validation, cada fold expande progressivamente a janela de treino mês a mês, simulando o cenário real de previsão.

Isso é fundamental porque técnicas como k-fold aleatório misturariam dados futuros e passados, inflando artificialmente as métricas.

---

## P4: Quais features têm maior importância preditiva?

**R:** As features mais importantes são, em ordem:

1. **Lags autoregressivos** (lag1, lag7) — O consumo recente é o melhor preditor do consumo futuro
2. **Médias móveis** (ma7, ma14) — Capturam a tendência de médio prazo
3. **Temperatura** — Correlação direta com uso de climatização
4. **Dia da semana / is_weekend** — Padrões semanais de uso doméstico
5. **Região** — Diferenças estruturais entre áreas geográficas

---

## P5: Por que foi usado DuckDB ao invés de Pandas puro para feature engineering?

**R:** O DuckDB oferece vantagens específicas para este caso:

- **SQL analítico nativo:** Funções de janela (`LAG`, `AVG OVER`, `STDDEV OVER`) são mais expressivas e legíveis para criar features autoregressivas do que o equivalente em Pandas
- **Performance:** DuckDB é otimizado para operações OLAP colunares, processando agregações mais rapidamente que Pandas em datasets maiores
- **Reprodutibilidade:** Queries SQL são auto-documentadas e facilmente auditáveis
- **Escalabilidade:** A mesma abordagem funcionaria com datasets significativamente maiores sem mudanças de código

---

## P6: Como os 5 clientes com região "Desconhecida" foram tratados?

**R:** Ao invés de simplesmente descartá-los (o que eliminaria 900 registros de consumo), foi adotada uma abordagem baseada em similaridade: o padrão de consumo desses clientes foi comparado com o padrão médio de cada região conhecida, e cada cliente foi atribuído à região cujo perfil de consumo mais se assemelhava ao seu. Isso preserva os dados enquanto mantém a integridade da variável regional.

---

## P7: O MAPE de ~14% é aceitável para o contexto de negócio?

**R:** Depende da aplicação:

- **Para planejamento de capacidade da rede elétrica:** 14% pode ser aceitável, pois erros individuais tendem a se cancelar quando agregados em muitos clientes
- **Para faturamento individual:** Seria insuficiente; seria necessário MAPE abaixo de 5%
- **Para detecção de anomalias (fraude/perdas):** Pode ser útil como baseline — desvios significativos da previsão sinalizariam consumo anormal
- **Para precificação dinâmica:** Razoável para projeções de médio prazo, mas necessitaria refinamento para previsões diárias precisas

---

## P8: Quais seriam os próximos passos recomendados para este projeto?

**R:** Evolução sugerida em três frentes:

**Dados:**
- Coletar mais histórico (mínimo 12 meses para capturar ciclo anual)
- Incorporar dados de feriados e eventos especiais
- Adicionar dados tarifários (preço da energia)

**Modelagem:**
- Testar modelos de séries temporais (Prophet, N-BEATS)
- Implementar ensemble stacking dos melhores modelos
- Otimizar hiperparâmetros com Optuna/Bayesian search

**Produção:**
- Containerizar com Docker para deploy
- Criar API REST (FastAPI) para previsões online
- Implementar monitoramento de drift de dados e modelo
- Automatizar retreinamento periódico (MLflow + Airflow)
