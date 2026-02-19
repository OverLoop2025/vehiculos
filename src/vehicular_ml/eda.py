"""Analisis exploratorio de datos (EDA) reproducible."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from vehicular_ml.data import ensure_target_column
from vehicular_ml.schema import NUMERIC_COLUMNS, TARGET_COLUMN

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


def run_eda(df: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    """Genera graficos y un resumen de EDA en Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_df = ensure_target_column(df)

    histogram_path = output_dir / "figura_1_distribucion_kilometraje.png"
    boxplot_path = output_dir / "figura_2_boxplot_temperatura.png"
    corr_path = output_dir / "figura_3_mapa_calor_correlacion.png"
    scatter_path = output_dir / "figura_4_scatter_kilometraje_falla.png"
    report_path = output_dir / "reporte_eda.md"

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(prepared_df["kilometraje"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribucion del kilometraje vehicular")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=prepared_df, y="temperatura_motor", ax=ax)
    ax.set_title("Valores atipicos en temperatura del motor")
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=140)
    plt.close(fig)

    corr_df = prepared_df[NUMERIC_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Mapa de calor de correlacion")
    fig.tight_layout()
    fig.savefig(corr_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=prepared_df,
        x="kilometraje",
        y="temperatura_motor",
        hue=TARGET_COLUMN,
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Relacion entre kilometraje y presencia de falla")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)

    nulls = prepared_df.isna().sum().sort_values(ascending=False)
    target_distribution = prepared_df[TARGET_COLUMN].value_counts(normalize=True).sort_index()

    report = f"""# Reporte EDA Vehicular

## Resumen del dataset
- Filas: {len(prepared_df)}
- Columnas: {prepared_df.shape[1]}
- Variable objetivo: `{TARGET_COLUMN}`

## Valores nulos por columna
{nulls.to_string()}

## Distribucion de la variable objetivo
{target_distribution.to_string()}

## Figuras generadas
1. `{histogram_path.name}`
2. `{boxplot_path.name}`
3. `{corr_path.name}`
4. `{scatter_path.name}`
"""

    report_path.write_text(report, encoding="utf-8")

    return {
        "histogram": str(histogram_path),
        "boxplot": str(boxplot_path),
        "heatmap": str(corr_path),
        "scatter": str(scatter_path),
        "report": str(report_path),
    }
