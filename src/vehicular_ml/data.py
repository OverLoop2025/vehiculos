"""Carga, validacion y preparacion de datos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from vehicular_ml.schema import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    build_target_column,
)


class DatasetValidationError(ValueError):
    """Error de validacion de esquema de datos."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un dataset CSV o Excel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise DatasetValidationError(
                f"Formato no soportado: {suffix}. Use CSV o Excel (.xlsx/.xls)."
            )
    except (EmptyDataError, ParserError) as exc:
        raise DatasetValidationError(f"Error al leer el archivo {path}: {exc}") from exc
    except ValueError as exc:
        raise DatasetValidationError(f"Formato invalido en {path}: {exc}") from exc

    if df.empty:
        raise DatasetValidationError("El dataset esta vacio.")

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Valida que el dataset contenga todas las columnas obligatorias."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise DatasetValidationError(
            "Faltan columnas requeridas: " + ", ".join(missing_columns)
        )


def normalize_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos para robustecer el pipeline de preprocesamiento."""
    output = df.copy()

    for column in NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        output[column] = output[column].astype(object).where(output[column].notna(), np.nan)

    return output


def _normalize_existing_target(target: pd.Series) -> pd.Series:
    normalized_target = pd.to_numeric(target, errors="coerce")
    if normalized_target.isna().any():
        raise DatasetValidationError(
            f"La columna {TARGET_COLUMN} contiene valores no numericos o nulos."
        )

    invalid_values = sorted(
        value for value in normalized_target.unique().tolist() if value not in {0, 1}
    )
    if invalid_values:
        raise DatasetValidationError(
            f"La columna {TARGET_COLUMN} debe ser binaria (0/1). "
            f"Valores detectados: {invalid_values}"
        )

    return normalized_target.astype(int)


def ensure_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura la variable objetivo en el dataframe."""
    validate_required_columns(df)

    output = normalize_feature_types(df)
    if TARGET_COLUMN not in output.columns:
        output[TARGET_COLUMN] = build_target_column(output)
    else:
        output[TARGET_COLUMN] = _normalize_existing_target(output[TARGET_COLUMN])
    return output


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y variable objetivo."""
    if TARGET_COLUMN not in df.columns:
        raise DatasetValidationError(f"No existe la columna objetivo: {TARGET_COLUMN}")

    x = df.drop(columns=[TARGET_COLUMN])
    y = _normalize_existing_target(df[TARGET_COLUMN])
    return x, y
