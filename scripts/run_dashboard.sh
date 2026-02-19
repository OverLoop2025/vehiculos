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
python -m streamlit run app.py
