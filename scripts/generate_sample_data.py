"""Genera un dataset sint'etico de mantenimiento vehicular."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    seed = 42
    rng = np.random.default_rng(seed)
    n_rows = 600

    marcas = np.array(["Toyota", "Hyundai", "Kia", "Chevrolet", "Nissan", "Suzuki"])
    combustibles = np.array(["Gasolina", "Di'esel", "GLP", "GNV"])
    aceites = np.array(["Sint'etico", "Semisint'etico", "Mineral"])
    fallas = np.array(["P0300", "P0420", "P0171", "P0562", "P0115", "P0016", "P0700"])

    kilometrage = rng.integers(10_000, 220_000, n_rows)
    temperature = np.clip(rng.normal(97, 9, n_rows), 75, 125).round(1)
    rpm = np.clip(rng.normal(2000, 450, n_rows), 700, 3500).round(0)
    usage_hours = np.clip(rng.normal(1800, 900, n_rows), 100, 5000).round(0)
    battery = np.clip(rng.normal(12.5, 0.8, n_rows), 10.5, 14.2).round(2)

    df = pd.DataFrame(
        {
            "kilometraje": kilometrage,
            "temperatura_motor": temperature,
            "rpm": rpm,
            "horas_uso": usage_hours,
            "voltaje_bateria": battery,
            "marca": rng.choice(marcas, size=n_rows, replace=True),
            "combustible": rng.choice(combustibles, size=n_rows, replace=True),
            "tipo_aceite": rng.choice(aceites, size=n_rows, replace=True),
            "codigo_falla": rng.choice(
                fallas,
                size=n_rows,
                p=np.array([0.18, 0.14, 0.16, 0.12, 0.12, 0.14, 0.14]),
                replace=True,
            ),
        }
    )

    needs_maintenance = (
        (df["kilometraje"] > 120_000)
        | (df["temperatura_motor"] > 105)
        | (df["voltaje_bateria"] < 12.0)
        | (df["codigo_falla"].isin(["P0300", "P0420"]))
    ).astype(int)

    noisy_flip = rng.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    df["requiere_mantenimiento"] = np.abs(needs_maintenance - noisy_flip)

    for column in ["temperatura_motor", "voltaje_bateria", "tipo_aceite"]:
        null_index = rng.choice(df.index, size=int(n_rows * 0.03), replace=False)
        df.loc[null_index, column] = np.nan

    output_path = Path("data/raw/vehicular_mantenimiento_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset generado en {output_path} con {len(df)} filas")


if __name__ == "__main__":
    main()
