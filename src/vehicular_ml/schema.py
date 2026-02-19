"""Esquema de datos y reglas de negocio para mantenimiento vehicular."""

from __future__ import annotations

from typing import Final

import pandas as pd

TARGET_COLUMN: Final[str] = "requiere_mantenimiento"

NUMERIC_COLUMNS: Final[list[str]] = [
    "kilometraje",
    "temperatura_motor",
    "rpm",
    "horas_uso",
    "voltaje_bateria",
]

CATEGORICAL_COLUMNS: Final[list[str]] = [
    "marca",
    "combustible",
    "tipo_aceite",
    "codigo_falla",
]

REQUIRED_COLUMNS: Final[list[str]] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

CRITICAL_FAILURE_CODES: Final[set[str]] = {"P0300", "P0420"}


def build_target_column(df: pd.DataFrame) -> pd.Series:
    """Genera la variable objetivo basada en reglas de mantenimiento preventivo."""
    return (
        (df["kilometraje"] > 120_000)
        | (df["temperatura_motor"] > 105)
        | (df["voltaje_bateria"] < 12.0)
        | (df["codigo_falla"].astype(str).isin(CRITICAL_FAILURE_CODES))
    ).astype(int)
