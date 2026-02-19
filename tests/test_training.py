from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vehicular_ml.training import TrainingConfig, TrainingError, train_and_save


def synthetic_training_df() -> pd.DataFrame:
    rows = []
    for i in range(100):
        rows.append(
            {
                "kilometraje": 30_000 + i * 2_000,
                "temperatura_motor": 88 + (i % 18),
                "rpm": 1300 + (i % 15) * 120,
                "horas_uso": 500 + i * 35,
                "voltaje_bateria": 13.4 - (i % 9) * 0.2,
                "marca": ["Toyota", "Hyundai", "Kia", "Chevrolet"][i % 4],
                "combustible": ["Gasolina", "Di'esel", "GLP"][i % 3],
                "tipo_aceite": ["Sint'etico", "Semisint'etico", "Mineral"][i % 3],
                "codigo_falla": ["P0171", "P0300", "P0420", "P0016", "P0562"][i % 5],
            }
        )
    return pd.DataFrame(rows)


def test_train_and_save_generates_artifacts(tmp_path: Path) -> None:
    df = synthetic_training_df()
    output_dir = tmp_path / "artifacts"

    result = train_and_save(
        df=df,
        output_dir=output_dir,
        config=TrainingConfig(test_size=0.25, random_state=11, n_estimators=50),
    )

    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "test_predictions.csv").exists()

    metrics = result["metrics"]
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        assert 0.0 <= metrics[metric_name] <= 1.0

    assert "created_at_utc" in result
    assert "confusion_matrix" in result
    assert "feature_importances_top" in result
    assert isinstance(result["feature_importances_top"], list)


def test_train_and_save_raises_if_target_has_single_class(tmp_path: Path) -> None:
    df = synthetic_training_df()
    df["requiere_mantenimiento"] = 1

    with pytest.raises(TrainingError):
        train_and_save(
            df=df,
            output_dir=tmp_path / "artifacts_single_class",
            config=TrainingConfig(test_size=0.2, random_state=7, n_estimators=25),
        )


def test_train_and_save_handles_null_categorical_values(tmp_path: Path) -> None:
    df = synthetic_training_df()
    df.loc[df.index[:8], "tipo_aceite"] = None

    result = train_and_save(
        df=df,
        output_dir=tmp_path / "artifacts_null_cat",
        config=TrainingConfig(test_size=0.2, random_state=13, n_estimators=30),
    )

    assert "metrics" in result
    assert (tmp_path / "artifacts_null_cat" / "model.joblib").exists()
