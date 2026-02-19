#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="vehicular-ml"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" && -z "${VEHICULAR_ML_CONDA_RUN:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    export VEHICULAR_ML_CONDA_RUN=1
    exec conda run -n "$ENV_NAME" bash "$0" "$@"
  fi
  echo "[WARN] Ejecutando sin entorno conda activo. Asegura dependencias instaladas."
fi

cd "$PROJECT_ROOT"

python scripts/generate_sample_data.py
python -m vehicular_ml.cli eda --data data/raw/vehicular_mantenimiento_sample.csv --output-dir artifacts/eda
python -m vehicular_ml.cli train --data data/raw/vehicular_mantenimiento_sample.csv --output-dir artifacts/model
python -m vehicular_ml.cli predict \
  --model artifacts/model/model.joblib \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output artifacts/predicciones.csv
python -m pytest -q

echo "[OK] Pipeline completo ejecutado correctamente."
