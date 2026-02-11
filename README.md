# Previsão de Consumo Energético Residencial

## Desafio Técnico - Cientista de Dados Pleno

Solução preditiva para estimar o consumo energético diário de clientes residenciais, utilizando dados históricos de consumo, variáveis climáticas e dados cadastrais.

---

## Estrutura do Projeto

```
├── data/                          # Dados brutos e processados
│   ├── consumo.csv                # Histórico de consumo por cliente/data
│   ├── clima.csv                  # Dados climáticos por região/data
│   ├── clientes.csv               # Dados cadastrais dos clientes
│   ├── energy.duckdb              # Banco DuckDB (gerado pelo pipeline)
│   ├── features_processed.parquet # Features processadas (gerado)
│   └── predictions.parquet        # Previsões do modelo (gerado)
├── notebooks/
│   ├── 01_eda.ipynb               # Análise Exploratória de Dados
│   └── 02_pipeline_modeling.ipynb # Pipeline, DuckDB, Features, Modelagem
├── dashboards/
│   └── dashboard.ipynb            # Dashboard interativo com Plotly
├── models/                        # Modelos treinados e metadados
│   ├── best_model.pkl             # Melhor modelo (gerado)
│   ├── random_forest_v*.pkl       # Random Forest versionado (gerado)
│   ├── lightgbm_v*.pkl            # LightGBM versionado (gerado)
│   ├── metadata_latest.json       # Metadados do último treinamento (gerado)
│   └── mlruns/                    # MLflow tracking (gerado)
├── src/
│   ├── pipeline.py                # Pipeline reprodutível (ingestão → inferência)
│   └── generate_presentation.py   # Gerador da apresentação PDF
├── apresentacao.pdf               # Apresentação executiva
├── requirements.txt               # Dependências Python
└── README.md                      # Este arquivo
```

---

## Tecnologias Utilizadas

| Tecnologia | Uso |
|------------|-----|
| **Python 3.9+** | Linguagem principal |
| **Pandas / NumPy** | Manipulação de dados |
| **DuckDB + SQL** | Banco analítico e feature engineering |
| **Scikit-learn** | Random Forest Regressor |
| **LightGBM** | Gradient Boosting Regressor |
| **MLflow** | Tracking de experimentos e modelos |
| **Plotly** | Dashboard interativo |
| **Matplotlib / Seaborn** | Visualizações na EDA |
| **Git** | Versionamento de código |

---

## Instalação e Configuração

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd desafio_aquarela_ia
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Ordem de Execução

### Opção 1: Pipeline Automatizado (Recomendado)

Executa todas as etapas de uma vez:

```bash
python src/pipeline.py
```

O pipeline executa sequencialmente:
1. **Ingestão** - Carrega os CSVs
2. **Transformação** - Limpa dados, cria tabelas DuckDB, gera features via SQL
3. **Treinamento** - Treina Random Forest e LightGBM com validação temporal
4. **Inferência** - Gera previsões com o melhor modelo

Para executar etapas individualmente:

```bash
python src/pipeline.py --step ingest
python src/pipeline.py --step transform
python src/pipeline.py --step train
python src/pipeline.py --step inference
```

### Opção 2: Notebooks Interativos

Execute na seguinte ordem:

1. **EDA**: `notebooks/01_eda.ipynb`
2. **Pipeline + Modelagem**: `notebooks/02_pipeline_modeling.ipynb`
3. **Dashboard**: `dashboards/dashboard.ipynb`

```bash
jupyter notebook
```

### Gerar Apresentação

```bash
python src/generate_presentation.py
```

---

## Sobre a Solução

### Tratamento de Dados

- **Regiões inconsistentes**: 5 clientes com região "Desconhecida" foram atribuídos à região com padrão de consumo mais similar
- **Temperaturas ausentes**: 45 valores (~5%) imputados por interpolação linear dentro de cada região

### Feature Engineering (via SQL no DuckDB)

- **Temporais**: mês, dia da semana, dia do mês, semana do ano, flag de fim de semana
- **Lags autoregressivos**: consumo dos últimos 1, 2, 3 e 7 dias
- **Médias móveis**: janelas de 7, 14 e 30 dias
- **Estatísticas móveis**: desvio padrão, mínimo e máximo (janela de 7 dias)
- **Climáticas**: temperatura, umidade, temperatura do dia anterior

### Modelos Treinados

| Modelo | MAE (kWh) | RMSE (kWh) | R² | MAPE (%) |
|--------|-----------|------------|-----|----------|
| Random Forest | 1.86 | 2.39 | 0.61 | 14.5 |
| **LightGBM** | **1.86** | **2.38** | **0.61** | **14.5** |

### Validação Temporal

- **Split principal**: Treino (Jan-Mai) / Teste (Jun)
- **Expanding Window CV**: Validação mês a mês com janela de treino crescente
- Evita data leakage temporal ao usar apenas dados passados para previsão

---

## Autor

Desenvolvido como parte do desafio técnico para a posição de Cientista de Dados Pleno.
