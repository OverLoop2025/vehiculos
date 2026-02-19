#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="vehicular-ml"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] No se encontro el comando 'conda'. Instala Miniconda."
  exit 1
fi

bash "$PROJECT_ROOT/scripts/bootstrap_conda.sh"
conda run -n "$ENV_NAME" bash "$PROJECT_ROOT/scripts/run_pipeline.sh"

echo "[OK] Inicio completo terminado en entorno '$ENV_NAME'."
