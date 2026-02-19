#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="vehicular-ml"
ENV_FILE="$PROJECT_ROOT/conda/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] No se encontro el comando 'conda'. Instala Miniconda primero."
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] No se encontro environment.yml en $ENV_FILE"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Helps fresh Miniconda installs running in non-interactive environments.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Actualizando entorno Conda existente: $ENV_NAME"
  conda env update -f "$ENV_FILE" --prune
else
  echo "[INFO] Creando entorno Conda: $ENV_NAME"
  conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT[dev]"

echo "[OK] Entorno listo."
echo "[OK] Activalo con: conda activate $ENV_NAME"
