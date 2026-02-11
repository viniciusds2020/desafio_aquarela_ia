"""
Pipeline de Previsão de Consumo Energético Residencial
=====================================================

Pipeline completo contendo:
1. Ingestão dos CSVs
2. Transformações (limpeza, feature engineering)
3. Treinamento
4. Inferência

Uso:
    python src/pipeline.py                    # Executa pipeline completo
    python src/pipeline.py --step ingest      # Apenas ingestão
    python src/pipeline.py --step transform   # Apenas transformação
    python src/pipeline.py --step train       # Apenas treinamento
    python src/pipeline.py --step inference   # Apenas inferência
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# ETAPA 1: INGESTÃO
# ============================================================
def ingest(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Carrega os CSVs brutos e retorna DataFrames."""
    logger.info("=== ETAPA 1: INGESTÃO ===")

    consumo = pd.read_csv(data_dir / "consumo.csv", parse_dates=["date"])
    clima = pd.read_csv(data_dir / "clima.csv", parse_dates=["date"])
    clientes = pd.read_csv(data_dir / "clientes.csv")

    logger.info(f"  consumo:  {consumo.shape}")
    logger.info(f"  clima:    {clima.shape}")
    logger.info(f"  clientes: {clientes.shape}")

    return {"consumo": consumo, "clima": clima, "clientes": clientes}


# ============================================================
# ETAPA 2: TRANSFORMAÇÕES
# ============================================================
def transform(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Limpa dados, cria tabelas DuckDB e gera features via SQL."""
    logger.info("=== ETAPA 2: TRANSFORMAÇÕES ===")

    consumo = data["consumo"]
    clima = data["clima"]
    clientes = data["clientes"]

    # --- 2a. Corrigir regiões inconsistentes ---
    logger.info("  Corrigindo regiões inconsistentes...")
    regioes_validas = sorted(clima.region.unique())
    consumo_merge = consumo.merge(clientes, on="client_id")
    media_por_regiao = (
        consumo_merge[consumo_merge.region != "Desconhecida"]
        .groupby("region")["consumption_kwh"]
        .mean()
    )

    clientes_corrigido = clientes.copy()
    for _, row in clientes[clientes.region == "Desconhecida"].iterrows():
        cid = row["client_id"]
        media_cliente = consumo[consumo.client_id == cid]["consumption_kwh"].mean()
        regiao_mais_proxima = (media_por_regiao - media_cliente).abs().idxmin()
        clientes_corrigido.loc[
            clientes_corrigido.client_id == cid, "region"
        ] = regiao_mais_proxima
        logger.info(
            f"    {cid}: média={media_cliente:.2f} -> {regiao_mais_proxima}"
        )

    # --- 2b. Imputar temperaturas ausentes ---
    logger.info("  Imputando temperaturas ausentes...")
    clima_tratado = clima.copy()
    n_missing_before = clima_tratado.temperature.isnull().sum()

    for regiao in clima_tratado.region.unique():
        mask = clima_tratado.region == regiao
        clima_tratado.loc[mask, "temperature"] = (
            clima_tratado.loc[mask, "temperature"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

    logger.info(
        f"    Temperaturas ausentes: {n_missing_before} -> {clima_tratado.temperature.isnull().sum()}"
    )

    # --- 2c. DuckDB: criar tabelas e gerar features ---
    logger.info("  Criando tabelas DuckDB e gerando features...")
    con = duckdb.connect(str(DATA_DIR / "energy.duckdb"))

    for tbl in ["consumo", "clima", "clientes"]:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")

    con.execute("CREATE TABLE clientes AS SELECT * FROM clientes_corrigido")
    con.execute("CREATE TABLE clima AS SELECT * FROM clima_tratado")
    con.execute("CREATE TABLE consumo AS SELECT * FROM consumo")

    query_features = """
    WITH consumo_com_regiao AS (
        SELECT co.client_id, co.date, co.consumption_kwh, cl.region
        FROM consumo co
        JOIN clientes cl ON co.client_id = cl.client_id
    ),
    consumo_com_clima AS (
        SELECT ccr.*, cli.temperature, cli.humidity
        FROM consumo_com_regiao ccr
        LEFT JOIN clima cli ON ccr.region = cli.region AND ccr.date = cli.date
    ),
    features_lag AS (
        SELECT *,
            EXTRACT(MONTH FROM date) AS month,
            EXTRACT(DOW FROM date) AS day_of_week,
            EXTRACT(DAY FROM date) AS day_of_month,
            EXTRACT(WEEK FROM date) AS week_of_year,
            CASE WHEN EXTRACT(DOW FROM date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
            LAG(consumption_kwh, 1) OVER (PARTITION BY client_id ORDER BY date) AS consumption_lag1,
            LAG(consumption_kwh, 2) OVER (PARTITION BY client_id ORDER BY date) AS consumption_lag2,
            LAG(consumption_kwh, 3) OVER (PARTITION BY client_id ORDER BY date) AS consumption_lag3,
            LAG(consumption_kwh, 7) OVER (PARTITION BY client_id ORDER BY date) AS consumption_lag7,
            AVG(consumption_kwh) OVER (
                PARTITION BY client_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ) AS consumption_ma7,
            AVG(consumption_kwh) OVER (
                PARTITION BY client_id ORDER BY date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
            ) AS consumption_ma14,
            AVG(consumption_kwh) OVER (
                PARTITION BY client_id ORDER BY date ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
            ) AS consumption_ma30,
            STDDEV(consumption_kwh) OVER (
                PARTITION BY client_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ) AS consumption_std7,
            MIN(consumption_kwh) OVER (
                PARTITION BY client_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ) AS consumption_min7,
            MAX(consumption_kwh) OVER (
                PARTITION BY client_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ) AS consumption_max7,
            LAG(temperature, 1) OVER (PARTITION BY client_id ORDER BY date) AS temp_lag1
        FROM consumo_com_clima
    )
    SELECT * FROM features_lag
    WHERE consumption_lag7 IS NOT NULL
    ORDER BY client_id, date
    """

    df_features = con.execute(query_features).fetchdf()
    con.close()

    # Encode região
    le = LabelEncoder()
    df_features["region_encoded"] = le.fit_transform(df_features["region"])

    # Dropar nulos residuais
    df_features = df_features.dropna()

    # Salvar label encoder
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Salvar features processadas
    df_features.to_parquet(DATA_DIR / "features_processed.parquet", index=False)

    logger.info(f"  Features geradas: {df_features.shape}")
    return df_features


# ============================================================
# ETAPA 3: TREINAMENTO
# ============================================================
FEATURE_COLS = [
    "temperature",
    "humidity",
    "month",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "consumption_lag1",
    "consumption_lag2",
    "consumption_lag3",
    "consumption_lag7",
    "consumption_ma7",
    "consumption_ma14",
    "consumption_ma30",
    "consumption_std7",
    "consumption_min7",
    "consumption_max7",
    "temp_lag1",
    "region_encoded",
]

TARGET = "consumption_kwh"


def evaluate(y_true, y_pred) -> dict:
    """Calcula métricas de avaliação."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def train(df_features: pd.DataFrame) -> dict:
    """Treina modelos Random Forest e LightGBM com validação temporal."""
    logger.info("=== ETAPA 3: TREINAMENTO ===")

    # Split temporal
    cutoff_date = pd.Timestamp("2023-06-01")
    train_mask = df_features["date"] < cutoff_date
    test_mask = df_features["date"] >= cutoff_date

    X_train = df_features.loc[train_mask, FEATURE_COLS]
    y_train = df_features.loc[train_mask, TARGET]
    X_test = df_features.loc[test_mask, FEATURE_COLS]
    y_test = df_features.loc[test_mask, TARGET]

    logger.info(f"  Treino: {X_train.shape[0]:,} amostras")
    logger.info(f"  Teste:  {X_test.shape[0]:,} amostras")

    # Configurar MLflow
    mlflow.set_tracking_uri(
        "file:///" + str(MODELS_DIR / "mlruns").replace("\\", "/")
    )
    mlflow.set_experiment("energy_consumption_prediction")

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # --- Random Forest ---
    logger.info("  Treinando Random Forest...")
    with mlflow.start_run(run_name="random_forest"):
        rf_params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)
        rf_metrics = evaluate(y_test, rf_pred)

        mlflow.log_params(rf_params)
        mlflow.log_metrics({f"test_{k}": v for k, v in rf_metrics.items()})
        mlflow.sklearn.log_model(rf_model, "random_forest_model")

    logger.info(f"    RF  -> MAE={rf_metrics['mae']:.4f}, R²={rf_metrics['r2']:.4f}")

    # --- LightGBM ---
    logger.info("  Treinando LightGBM...")
    with mlflow.start_run(run_name="lightgbm"):
        lgb_params = {
            "n_estimators": 300,
            "max_depth": 10,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train)

        lgb_pred = lgb_model.predict(X_test)
        lgb_metrics = evaluate(y_test, lgb_pred)

        mlflow.log_params(lgb_params)
        mlflow.log_metrics({f"test_{k}": v for k, v in lgb_metrics.items()})
        mlflow.sklearn.log_model(lgb_model, "lightgbm_model")

    logger.info(
        f"    LGB -> MAE={lgb_metrics['mae']:.4f}, R²={lgb_metrics['r2']:.4f}"
    )

    # Salvar modelos
    with open(MODELS_DIR / f"random_forest_v{version}.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(MODELS_DIR / f"lightgbm_v{version}.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    # Determinar melhor modelo
    best_name = "lightgbm" if lgb_metrics["rmse"] < rf_metrics["rmse"] else "random_forest"
    best_model = lgb_model if best_name == "lightgbm" else rf_model

    # Salvar melhor modelo como "best"
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Metadados
    metadata = {
        "version": version,
        "best_model": best_name,
        "feature_cols": FEATURE_COLS,
        "target": TARGET,
        "metrics": {"random_forest": rf_metrics, "lightgbm": lgb_metrics},
    }
    with open(MODELS_DIR / f"metadata_v{version}.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(MODELS_DIR / "metadata_latest.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Melhor modelo: {best_name}")
    logger.info(f"  Modelos salvos em {MODELS_DIR}")

    # Salvar previsões
    predictions = df_features.loc[test_mask].copy()
    predictions["rf_prediction"] = rf_pred
    predictions["lgb_prediction"] = lgb_pred
    predictions.to_parquet(DATA_DIR / "predictions.parquet", index=False)

    return metadata


# ============================================================
# ETAPA 4: INFERÊNCIA
# ============================================================
def inference(
    df_input: pd.DataFrame = None, model_path: Path = None
) -> pd.DataFrame:
    """Executa inferência com o melhor modelo salvo."""
    logger.info("=== ETAPA 4: INFERÊNCIA ===")

    if model_path is None:
        model_path = MODELS_DIR / "best_model.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"  Modelo carregado: {model_path}")

    if df_input is None:
        df_input = pd.read_parquet(DATA_DIR / "features_processed.parquet")
        logger.info(f"  Dados carregados: {df_input.shape}")

    predictions = model.predict(df_input[FEATURE_COLS])
    df_input = df_input.copy()
    df_input["prediction"] = predictions

    output_path = DATA_DIR / "inference_output.parquet"
    df_input.to_parquet(output_path, index=False)
    logger.info(f"  Inferência salva: {output_path}")
    logger.info(
        f"  Previsão média: {predictions.mean():.2f} kWh"
    )

    return df_input


# ============================================================
# PIPELINE COMPLETO
# ============================================================
def run_pipeline():
    """Executa o pipeline completo: ingestão -> transformação -> treinamento -> inferência."""
    logger.info("=" * 60)
    logger.info("PIPELINE DE PREVISÃO DE CONSUMO ENERGÉTICO")
    logger.info("=" * 60)

    data = ingest()
    df_features = transform(data)
    metadata = train(df_features)
    df_output = inference()

    logger.info("=" * 60)
    logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
    logger.info("=" * 60)

    return df_output, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de Previsão de Consumo Energético"
    )
    parser.add_argument(
        "--step",
        choices=["ingest", "transform", "train", "inference", "all"],
        default="all",
        help="Etapa do pipeline a executar (default: all)",
    )
    args = parser.parse_args()

    if args.step == "all":
        run_pipeline()
    elif args.step == "ingest":
        ingest()
    elif args.step == "transform":
        data = ingest()
        transform(data)
    elif args.step == "train":
        df = pd.read_parquet(DATA_DIR / "features_processed.parquet")
        train(df)
    elif args.step == "inference":
        inference()
