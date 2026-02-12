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

**R:** O **LightGBM** foi selecionado como melhor modelo, embora a diferença seja marginal:

| Métrica | Random Forest | LightGBM | Diferença |
|---------|---------------|----------|-----------|
| MAE | 1.8620 kWh | **1.8607 kWh** | -0.0013 |
| RMSE | 2.3308 kWh | **2.3257 kWh** | -0.0051 |
| R² | 0.6117 | **0.6134** | +0.0017 |
| MAPE | 13.91% | **13.81%** | -0.10 pp |

O LightGBM vence em todas as métricas, mas por margens mínimas. As razões práticas para a escolha vão além das métricas:

- **Tamanho do modelo:** O LightGBM serializado ocupa ~895 KB, enquanto o Random Forest ocupa ~52 MB — quase **60x menor**, facilitando deploy e versionamento.
- **Eficiência:** Gradient Boosting constrói árvores sequenciais que corrigem erros das anteriores, convergindo com menos árvores (300) do que o RF precisa (200 árvores completas, porém mais profundas).
- **Estabilidade na validação cruzada temporal:** Ao longo dos 4 folds expansíveis (Mar→Jun), o LightGBM manteve MAE de **1.8585 ± 0.0063 kWh**, demonstrando baixa variância e boa capacidade de generalização.

---

## P2: O R² de ~0.61 é satisfatório? Como poderia ser melhorado?

**R:** Um R² de 0.6134 indica que o modelo explica **61,3% da variabilidade** do consumo diário. Para avaliar se isso é satisfatório, é preciso considerar os dados disponíveis:

**Contexto dos dados:**
- Consumo médio: **14.81 kWh** com desvio padrão de **3.73 kWh**
- MAE de 1.86 kWh representa ~12.6% do consumo médio
- Coeficiente de variação dos clientes: **0.19** (19% de variabilidade intra-cliente)

**Por que o R² não é mais alto:**
- Apenas **6 meses** de dados — insuficiente para capturar sazonalidade anual completa
- Correlação clima × consumo é **fraca** (entre -0.036 e -0.075 por região)
- Variação por dia da semana é **quase inexistente** (14.77–14.86 kWh, diferença de apenas 0.09 kWh entre o dia de maior e menor consumo)
- Ausência de variáveis explicativas importantes (tarifas, feriados, composição familiar)

**Como melhorar:**
1. **Mais histórico** — No mínimo 12 meses para capturar ciclo anual (inverno vs. verão). O R² provavelmente subiria com sazonalidade completa.
2. **Features externas** — Feriados, preço da energia, dados socioeconômicos por região.
3. **Modelos de séries temporais** — Prophet ou N-BEATS, que modelam tendência e sazonalidade explicitamente.
4. **Otimização de hiperparâmetros** — Optuna/Bayesian search ao invés dos valores fixos atuais (max_depth=10, lr=0.05).
5. **Ensemble stacking** — Combinar RF + LightGBM + modelo temporal para capturar padrões complementares.

---

## P3: Como o pipeline lida com o vazamento temporal (data leakage)?

**R:** O pipeline implementa **duas camadas de proteção** contra data leakage:

**1. Divisão temporal estrita:**
- Treino: 2023-01-08 a 2023-05-31 → **14.400 amostras**
- Teste: 2023-06-01 a 2023-06-29 → **2.900 amostras**
- O corte é definido com `cutoff_date = '2023-06-01'` no `pipeline.py`
- Nenhuma informação de Junho participa do treinamento

**2. Validação cruzada com janela expansível (Expanding Window CV):**

| Fold | Período Treino | Período Teste | Amostras Treino | Amostras Teste | R² |
|------|----------------|---------------|-----------------|----------------|----|
| 1 | Jan–Fev | Março | 5.200 | 3.100 | 0.6025 |
| 2 | Jan–Mar | Abril | 8.300 | 3.000 | 0.6077 |
| 3 | Jan–Abr | Maio | 11.300 | 3.100 | 0.6191 |
| 4 | Jan–Mai | Junho | 14.400 | 2.900 | 0.6134 |

Isso simula o cenário real: treinar com dados passados e prever o mês seguinte. A consistência dos resultados (R² de 0.6025 a 0.6191, com média **0.6107 ± 0.0072**) confirma que o modelo **não está overfitting** — ele generaliza de forma estável ao longo do tempo.

**Por que isso importa:** Um k-fold aleatório teria lags do dia 5 de Junho no treino e tentaria prever o dia 4 de Junho, usando informação do futuro. A validação temporal evita completamente essa contaminação.

---

## P4: Quais features têm maior importância preditiva?

**R:** Com base na importância de features extraída dos modelos treinados (19 features no total), o ranking organiza-se em tiers:

**Tier 1 — Dominantes (features autoregressivas):**
- `consumption_lag1` — Consumo do dia anterior (preditor mais forte)
- `consumption_lag7` — Consumo do mesmo dia da semana anterior
- `consumption_ma7` — Média móvel de 7 dias
- `consumption_ma14` — Média móvel de 14 dias

**Tier 2 — Significativas:**
- `consumption_lag2`, `consumption_lag3` — Lags recentes
- `consumption_ma30` — Tendência de longo prazo
- `consumption_std7` — Volatilidade recente do consumo
- `consumption_min7`, `consumption_max7` — Range de consumo na semana

**Tier 3 — Contribuição secundária:**
- `temperature`, `temp_lag1` — Variáveis climáticas (correlação fraca: -0.036 a -0.075 por região)
- `day_of_week`, `month` — Componentes temporais
- `region_encoded` — Região geográfica

**Tier 4 — Contribuição marginal:**
- `humidity` — Umidade relativa
- `is_weekend`, `day_of_month`, `week_of_year`

**Insight principal:** O consumo energético é fundamentalmente um **processo autoregressivo** — o melhor preditor do consumo de amanhã é o consumo de hoje e dos dias anteriores. As variáveis climáticas, embora estatisticamente significativas, têm impacto secundário neste dataset de 6 meses sem extremos climáticos fortes.

---

## P5: Por que foi usado DuckDB ao invés de Pandas puro para feature engineering?

**R:** O DuckDB trouxe vantagens concretas neste projeto em 5 dimensões:

**1. Expressividade SQL para funções de janela:**
As 20 features foram criadas com SQL declarativo — `LAG()`, `AVG() OVER()`, `STDDEV() OVER()` — mais claro e menos propenso a erros do que o equivalente em Pandas (`groupby().shift()`, `groupby().rolling().mean()`).

**2. Performance em operações analíticas:**
DuckDB é um motor OLAP colunar otimizado para agregações — mais eficiente que Pandas para window functions sobre 18.000 registros particionados por 100 clientes.

**3. Auditabilidade:**
As queries SQL no `pipeline.py` são auto-documentadas. Um analista de dados pode auditar a lógica de feature engineering sem conhecer a API do Pandas.

**4. Escalabilidade futura:**
Com 18.000 registros, Pandas seria suficiente. Mas se o dataset crescer para milhões (mais clientes, granularidade horária), o DuckDB escala sem mudanças de código. A mesma query SQL funcionaria com 100x mais dados.

**5. Integração com armazenamento persistente:**
O DuckDB persiste os dados em `data/energy.duckdb`, servindo como banco analítico local que pode ser consultado independentemente do pipeline Python.

---

## P6: Como os 5 clientes com região "Desconhecida" foram tratados?

**R:** Os 5 clientes sem região foram reatribuídos por **similaridade de padrão de consumo**, ao invés de descartados (o que eliminaria 5 × 180 = **900 registros**, 5% do dataset):

| Cliente | Consumo Médio (kWh) | Região Atribuída |
|---------|---------------------|------------------|
| C0015 | 17.49 | Oeste |
| C0050 | 15.16 | Leste |
| C0058 | 12.19 | Norte |
| C0077 | 16.41 | Oeste |
| C0092 | 11.78 | Norte |

**Metodologia:** O consumo médio de cada cliente "Desconhecido" foi comparado com o perfil de consumo das 5 regiões conhecidas:

| Região | Consumo Médio | Desvio Padrão | N. Clientes |
|--------|---------------|---------------|-------------|
| Oeste | 15.96 kWh | 3.49 | 15 |
| Leste | 14.98 kWh | 4.04 | 18 |
| Centro | 14.95 kWh | 3.27 | 16 |
| Sul | 14.69 kWh | 3.55 | 15 |
| Norte | 14.25 kWh | 3.62 | 31 |

**Validação:** As atribuições são consistentes — C0058 (12.19 kWh) e C0092 (11.78 kWh) foram ao Norte (menor média: 14.25 kWh), enquanto C0015 (17.49 kWh) e C0077 (16.41 kWh) foram ao Oeste (maior média: 15.96 kWh). A lógica reflete corretamente os padrões regionais de consumo.

---

## P7: O MAPE de ~14% é aceitável para o contexto de negócio?

**R:** O MAPE de **13.81%** (LightGBM) precisa ser avaliado conforme o caso de uso específico:

**Aceitável para:**

| Caso de Uso | Justificativa |
|-------------|---------------|
| **Planejamento de rede** | Quando se agrega previsão de muitos clientes, erros individuais se cancelam. Com 100 clientes, o erro agregado seria significativamente menor que 14%. |
| **Dimensionamento de infraestrutura** | Para decisões de médio/longo prazo sobre expansão de rede, 14% é razoável. |
| **Detecção de anomalias** | O modelo serve como baseline — se um cliente consumir 40% acima da previsão, isso sinaliza possível fraude, defeito ou mudança de hábito. |
| **Segmentação de clientes** | A previsão identifica padrões regionais e perfis úteis para marketing e tarifação. |

**Insuficiente para:**

| Caso de Uso | Requisito Típico |
|-------------|-----------------|
| **Faturamento individual** | MAPE < 5% |
| **Precificação horária dinâmica** | Granularidade e precisão muito maiores |
| **Contratos de fornecimento** | Margens de erro menores para compromissos contratuais |

**Contextualizando os números:**
- MAE de 1.86 kWh sobre consumo médio de 14.81 kWh
- O maior consumidor (C0030, região Leste) consome **19.38 kWh** e o menor (C0029) consome **9.40 kWh** — faixa de ~10 kWh
- O erro médio (1.86 kWh) representa menos de 20% dessa faixa inter-clientes

---

## P8: Quais seriam os próximos passos recomendados para este projeto?

**R:** Evolução recomendada em três frentes, priorizadas por impacto esperado:

### Dados (maior impacto potencial)
- **Ampliar histórico para 12+ meses** — Apenas 6 meses impossibilita capturar sazonalidade anual. O R² provavelmente subiria com dados de verão + inverno completos.
- **Incorporar feriados e eventos** — A variação por dia da semana é quase zero (14.77–14.86 kWh), mas feriados podem ter impacto real mascarado.
- **Dados tarifários** — Preço da energia pode influenciar comportamento de consumo (elasticidade-preço).
- **Granularidade horária** — Permitiria capturar picos de demanda e padrões intra-dia.

### Modelagem
- **Otimização de hiperparâmetros com Optuna** — Os parâmetros atuais são fixos (max_depth=10, lr=0.05, n_estimators=300); busca bayesiana pode encontrar combinações superiores.
- **Modelos de séries temporais** — Prophet ou N-BEATS capturam tendência e sazonalidade explicitamente, o que árvores de decisão não fazem.
- **Ensemble stacking** — Combinar LightGBM + RF + modelo temporal, usando previsões de cada um como features para um meta-modelo.
- **Features de interação** — Temperatura × região, dia da semana × região — podem revelar padrões condicionais.

### Produção
- **Containerização Docker** — Garantir reprodutibilidade em qualquer ambiente de deploy.
- **API REST com FastAPI** — Expor o modelo como serviço para previsões online.
- **Monitoramento de drift** — Detectar quando a distribuição dos dados muda e o modelo precisa ser retreinado.
- **Retreinamento automatizado** — MLflow + Airflow para pipeline de retreinamento periódico (ex: mensal).
- **Testes automatizados** — Validar que novas versões do modelo não degradam métricas abaixo de thresholds mínimos.
