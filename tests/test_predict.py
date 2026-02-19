from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vehicular_ml.data import DatasetValidationError
from vehicular_ml.predict import predict_dataframe
from vehicular_ml.training import TrainingConfig, train_and_save


def dataset() -> pd.DataFrame:
    rows = []
    for i in range(80):
        rows.append(
            {
                "kilometraje": 20_000 + i * 2_300,
                "temperatura_motor": 85 + (i % 20),
                "rpm": 1200 + (i % 16) * 110,
                "horas_uso": 300 + i * 40,
                "voltaje_bateria": 13.5 - (i % 10) * 0.22,
                "marca": ["Toyota", "Hyundai", "Kia", "Chevrolet"][i % 4],
                "combustible": ["Gasolina", "Di'esel", "GLP"][i % 3],
                "tipo_aceite": ["Sint'etico", "Semisint'etico", "Mineral"][i % 3],
                "codigo_falla": ["P0171", "P0300", "P0420", "P0016", "P0115"][i % 5],
            }
        )
    return pd.DataFrame(rows)


def test_predict_dataframe_outputs_probability_and_class(tmp_path: Path) -> None:
    df = dataset()
    output_dir = tmp_path / "train"

    train_and_save(
        df=df,
        output_dir=output_dir,
        config=TrainingConfig(test_size=0.2, random_state=3, n_estimators=40),
    )

    prediction_df = predict_dataframe(output_dir / "model.joblib", df.head(12))

    assert "prediccion_mantenimiento" in prediction_df.columns
    assert "probabilidad_mantenimiento" in prediction_df.columns
    assert len(prediction_df) == 12
    assert ((prediction_df["probabilidad_mantenimiento"] >= 0.0) & (prediction_df["probabilidad_mantenimiento"] <= 1.0)).all()


def test_predict_dataframe_raises_for_incomplete_schema(tmp_path: Path) -> None:
    df = dataset()
    output_dir = tmp_path / "train_schema"

    train_and_save(
        df=df,
        output_dir=output_dir,
        config=TrainingConfig(test_size=0.2, random_state=5, n_estimators=30),
    )

    broken_df = df.drop(columns=["codigo_falla"]).head(5)

    with pytest.raises(DatasetValidationError):
        predict_dataframe(output_dir / "model.joblib", broken_df)
