from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vehicular_ml.data import DatasetValidationError, ensure_target_column, load_dataset


def base_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kilometraje": [45_000, 150_000, 95_000, 130_000],
            "temperatura_motor": [90, 110, 98, 104],
            "rpm": [1500, 2500, 1900, 2300],
            "horas_uso": [800, 3200, 1700, 2500],
            "voltaje_bateria": [13.2, 11.7, 12.5, 11.9],
            "marca": ["Toyota", "Hyundai", "Kia", "Nissan"],
            "combustible": ["Gasolina", "Di'esel", "Gasolina", "GLP"],
            "tipo_aceite": ["Sint'etico", "Semisint'etico", "Sint'etico", "Mineral"],
            "codigo_falla": ["P0171", "P0300", "P0016", "P0562"],
        }
    )


def test_load_dataset_csv_success(tmp_path: Path) -> None:
    df = base_dataframe()
    csv_path = tmp_path / "vehiculos.csv"
    df.to_csv(csv_path, index=False)

    loaded_df = load_dataset(csv_path)

    assert len(loaded_df) == len(df)
    assert set(loaded_df.columns) == set(df.columns)


def test_load_dataset_raises_for_unsupported_format(tmp_path: Path) -> None:
    bad_file = tmp_path / "vehiculos.txt"
    bad_file.write_text("not,a,valid,format", encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_dataset(bad_file)


def test_ensure_target_column_generates_binary_target() -> None:
    df = base_dataframe()

    result = ensure_target_column(df)

    assert "requiere_mantenimiento" in result.columns
    assert set(result["requiere_mantenimiento"].unique()).issubset({0, 1})


def test_ensure_target_column_raises_if_schema_is_incomplete() -> None:
    df = base_dataframe().drop(columns=["codigo_falla"])

    with pytest.raises(DatasetValidationError):
        ensure_target_column(df)


def test_ensure_target_column_raises_if_existing_target_is_invalid() -> None:
    df = base_dataframe()
    df["requiere_mantenimiento"] = [1, 0, 2, 1]

    with pytest.raises(DatasetValidationError):
        ensure_target_column(df)
