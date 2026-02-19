"""Entrenamiento, evaluacion y persistencia del modelo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from vehicular_ml.data import ensure_target_column, split_features_target
from vehicular_ml.preprocessing import build_preprocessor


class TrainingError(ValueError):
    """Error de validacion o ejecucion de entrenamiento."""


@dataclass(slots=True)
class TrainingConfig:
    """Configuracion principal de entrenamiento."""

    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300


def _validate_training_config(config: TrainingConfig) -> None:
    if not 0.0 < config.test_size < 1.0:
        raise TrainingError("`test_size` debe estar entre 0 y 1.")
    if config.n_estimators <= 0:
        raise TrainingError("`n_estimators` debe ser mayor que 0.")


def _build_model_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    random_state=config.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _compute_metrics(y_true: pd.Series, y_pred: Any, y_prob: Any | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass
    return metrics


def train_and_save(
    df: pd.DataFrame,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Entrena el modelo y guarda artefactos reproducibles."""
    config = config or TrainingConfig()
    _validate_training_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_df = ensure_target_column(df)
    x, y = split_features_target(prepared_df)

    class_counts = y.value_counts().sort_index()
    if len(class_counts) < 2:
        raise TrainingError(
            "La variable objetivo solo contiene una clase. "
            "Se requieren al menos dos clases para entrenar."
        )
    if class_counts.min() < 2:
        raise TrainingError(
            "Cada clase debe tener al menos 2 registros para hacer split estratificado."
        )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    except ValueError as exc:
        raise TrainingError(
            "No fue posible dividir los datos en train/test. "
            "Revise `test_size` y la distribucion del target."
        ) from exc

    pipeline = _build_model_pipeline(config)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob: Any | None = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = _compute_metrics(y_test, y_pred, y_prob)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"

    joblib.dump(pipeline, model_path)

    result_df = x_test.copy()
    result_df["y_true"] = y_test.values
    result_df["y_pred"] = y_pred
    result_df["y_prob"] = y_prob if y_prob is not None else float("nan")
    result_df.to_csv(predictions_path, index=False)

    payload = {
        "config": asdict(config),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "metrics": metrics,
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "test_predictions": str(predictions_path),
        },
    }

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
