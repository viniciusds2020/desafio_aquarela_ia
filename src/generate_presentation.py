"""
Gerador da Apresentação Executiva em PDF
=========================================

Gera o arquivo apresentacao.pdf com storytelling sobre o projeto.
"""

import json
import os
import sys
from pathlib import Path

from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class Presentation(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Desafio Tecnico - Cientista de Dados Pleno", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Pagina {self.page_no()}/{{nb}}", align="C")

    def slide_title(self, title, subtitle=""):
        self.add_page()
        self.ln(60)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(30, 60, 114)
        self.cell(0, 15, title, align="C")
        if subtitle:
            self.ln(20)
            self.set_font("Helvetica", "", 16)
            self.set_text_color(80, 80, 80)
            self.cell(0, 10, subtitle, align="C")

    def slide_content(self, title, items):
        self.add_page()
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(30, 60, 114)
        self.cell(0, 15, title, align="L")
        self.ln(20)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(50, 50, 50)
        for item in items:
            if item.startswith("##"):
                self.ln(5)
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(30, 60, 114)
                self.cell(0, 8, item.replace("## ", ""), align="L")
                self.ln(8)
                self.set_font("Helvetica", "", 13)
                self.set_text_color(50, 50, 50)
            elif item.startswith("- "):
                self.cell(10)
                self.cell(0, 8, f"  {item}", align="L")
                self.ln(8)
            else:
                self.multi_cell(0, 8, item)
                self.ln(3)

    def slide_metrics(self, title, metrics_data):
        self.add_page()
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(30, 60, 114)
        self.cell(0, 15, title, align="L")
        self.ln(25)

        # Table header
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(30, 60, 114)
        self.set_text_color(255, 255, 255)
        col_w = [60, 55, 55]
        headers = ["Metrica", "Random Forest", "LightGBM"]
        for i, h in enumerate(headers):
            self.cell(col_w[i], 12, h, border=1, fill=True, align="C")
        self.ln()

        # Table rows
        self.set_font("Helvetica", "", 12)
        self.set_text_color(50, 50, 50)
        for row in metrics_data:
            fill = False
            self.set_fill_color(240, 245, 255)
            for i, val in enumerate(row):
                self.cell(col_w[i], 10, str(val), border=1, fill=fill, align="C")
            self.ln()


def generate():
    # Load metadata
    meta_path = MODELS_DIR / "metadata_latest.json"
    if not meta_path.exists():
        meta_files = sorted(MODELS_DIR.glob("metadata_v*.json"))
        if meta_files:
            meta_path = meta_files[-1]
        else:
            print("ERRO: Nenhum arquivo de metadados encontrado. Execute o pipeline primeiro.")
            sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    rf = metadata["metrics"]["random_forest"]
    lgb_m = metadata["metrics"]["lightgbm"]
    best = metadata["best_model"]

    pdf = Presentation("L", "mm", "A4")  # Landscape
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Slide 1: Capa
    pdf.slide_title(
        "Previsao de Consumo Energetico",
        "Desafio Tecnico - Cientista de Dados Pleno"
    )
    pdf.ln(30)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Solucao preditiva para consumo energetico residencial", align="C")

    # Slide 2: Contexto e Objetivo
    pdf.slide_content("Contexto e Objetivo", [
        "## Contexto",
        "- Dados de 100 clientes residenciais ao longo de 180 dias (Jan-Jun 2023)",
        "- Dados climaticos de 5 regioes (temperatura e umidade)",
        "- Desafio: estimar o consumo energetico futuro dos clientes",
        "",
        "## Objetivo",
        "- Construir modelo preditivo para consumo energetico diario",
        "- Identificar fatores que influenciam o consumo",
        "- Criar pipeline reprodutivel e escalavel",
    ])

    # Slide 3: Dados
    pdf.slide_content("Dados Disponveis", [
        "## Datasets",
        "- consumo.csv: 18.000 registros (100 clientes x 180 dias)",
        "- clima.csv: 900 registros (5 regioes x 180 dias) - 45 temperaturas ausentes",
        "- clientes.csv: 100 clientes - 5 com regiao 'Desconhecida'",
        "",
        "## Tratamento Realizado",
        "- Regioes 'Desconhecida': atribuidas por similaridade de consumo",
        "- Temperaturas ausentes: interpolacao linear por regiao",
        "- Tabelas normalizadas criadas no DuckDB",
    ])

    # Slide 4: Feature Engineering
    pdf.slide_content("Feature Engineering", [
        "## Features Temporais",
        "- Mes, dia da semana, dia do mes, semana do ano, flag weekend",
        "",
        "## Features de Lag (autoregressivas)",
        "- Consumo dos ultimos 1, 2, 3 e 7 dias",
        "- Medias moveis de 7, 14 e 30 dias",
        "- Desvio padrao movel de 7 dias",
        "- Min/Max movel de 7 dias",
        "",
        "## Features Climaticas",
        "- Temperatura e umidade do dia",
        "- Temperatura do dia anterior (lag)",
        "- Regiao codificada (Label Encoding)",
    ])

    # Slide 5: Modelagem
    pdf.slide_content("Modelagem", [
        "## Modelos Treinados",
        "- Random Forest Regressor (200 arvores, max_depth=15)",
        "- LightGBM Regressor (300 iteracoes, learning_rate=0.05)",
        "",
        "## Validacao Temporal",
        "- Split temporal: treino Jan-Mai / teste Jun",
        "- Expanding window cross-validation (Mar a Jun)",
        "- Evita data leakage temporal",
        "",
        "## Registro de Modelos",
        "- MLflow para tracking de experimentos",
        "- Pickle versionado para deploy",
    ])

    # Slide 6: Resultados
    metrics_table = [
        ["MAE (kWh)", f"{rf['mae']:.4f}", f"{lgb_m['mae']:.4f}"],
        ["RMSE (kWh)", f"{rf['rmse']:.4f}", f"{lgb_m['rmse']:.4f}"],
        ["R2", f"{rf['r2']:.4f}", f"{lgb_m['r2']:.4f}"],
        ["MAPE (%)", f"{rf['mape']:.2f}", f"{lgb_m['mape']:.2f}"],
    ]
    pdf.slide_metrics("Resultados - Metricas no Teste", metrics_table)
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 120, 60)
    best_label = "LightGBM" if best == "lightgbm" else "Random Forest"
    pdf.cell(0, 10, f"Melhor modelo: {best_label}", align="L")

    # Slide 7: Insights
    pdf.slide_content("Principais Insights", [
        "## Features Mais Importantes",
        "- Lags de consumo (1 e 7 dias) sao os preditores mais fortes",
        "- Medias moveis capturam tendencias de medio prazo",
        "- Variaveis climaticas tem impacto secundario",
        "",
        "## Padroes Identificados",
        "- Consumo varia sazonalmente (verao -> inverno)",
        "- Padroes semanais presentes (dias uteis vs fim de semana)",
        "- Regioes apresentam niveis de consumo distintos",
        "",
        "## Qualidade dos Dados",
        "- 5% das temperaturas ausentes (tratadas por interpolacao)",
        "- 5 clientes com regiao incorreta (corrigidos por similaridade)",
    ])

    # Slide 8: Pipeline e Arquitetura
    pdf.slide_content("Pipeline e Arquitetura", [
        "## Pipeline Reprodutivel (src/pipeline.py)",
        "- Etapa 1: Ingestao dos CSVs",
        "- Etapa 2: Transformacoes (limpeza, DuckDB, feature engineering SQL)",
        "- Etapa 3: Treinamento (RF + LightGBM com MLflow)",
        "- Etapa 4: Inferencia (previsoes com melhor modelo)",
        "",
        "## Tecnologias Utilizadas",
        "- Python (Pandas, Scikit-learn, LightGBM)",
        "- DuckDB + SQL para feature engineering",
        "- MLflow para tracking de experimentos",
        "- Plotly para dashboard interativo",
        "- Git para versionamento",
    ])

    # Slide 9: Proximos Passos
    pdf.slide_content("Proximos Passos", [
        "## Melhorias no Modelo",
        "- Hyperparameter tuning com Optuna/Bayesian Search",
        "- Modelos de series temporais (Prophet, LSTM)",
        "- Ensemble dos modelos (stacking)",
        "",
        "## Engenharia",
        "- Containerizacao com Docker",
        "- CI/CD para retreinamento automatico",
        "- Monitoramento de drift de dados e modelo",
        "",
        "## Dados",
        "- Incorporar dados de feriados e eventos",
        "- Dados de tarifa energetica",
        "- Dados socioeconomicos por regiao",
    ])

    # Slide 10: Obrigado
    pdf.slide_title("Obrigado!", "")

    output_path = PROJECT_ROOT / "apresentacao.pdf"
    pdf.output(str(output_path))
    print(f"Apresentacao gerada: {output_path}")


if __name__ == "__main__":
    generate()
