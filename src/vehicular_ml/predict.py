"""Inferencia sobre nuevos datos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from vehicular_ml.data import normalize_feature_types, validate_required_columns
from vehicular_ml.schema import TARGET_COLUMN


class PredictionError(ValueError):
    """Error de validacion o ejecucion de prediccion."""


def _extract_probabilities(model: Any, x: pd.DataFrame) -> Any:
    if not hasattr(model, "predict_proba"):
        raise PredictionError(
            "El modelo cargado no soporta `predict_proba`; no se puede calcular probabilidad."
        )

    probabilities = model.predict_proba(x)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise PredictionError("Formato invalido de probabilidades devuelto por el modelo.")

    if probabilities.shape[1] == 1:
        return probabilities[:, 0]

    return probabilities[:, 1]


def predict_dataframe(model_path: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """Genera predicciones y probabilidades para un dataframe."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el artefacto de modelo: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"No se pudo cargar el modelo en {model_path}: {exc}") from exc

    input_df = df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    validate_required_columns(input_df)
    input_df = normalize_feature_types(input_df)

    try:
        y_pred = model.predict(input_df)
        y_prob = _extract_probabilities(model, input_df)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Error durante la prediccion: {exc}") from exc

    output = input_df.copy()
    output["prediccion_mantenimiento"] = y_pred
    output["probabilidad_mantenimiento"] = y_prob
    return output
