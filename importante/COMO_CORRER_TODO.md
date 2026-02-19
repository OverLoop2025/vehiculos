# Como Correr Todo el Proyecto Correctamente

Guia oficial de ejecucion local para el proyecto `vehicular_ml`.

## 1) Requisitos previos

- Tener Miniconda o Anaconda instalado.
- Estar en la raiz del proyecto:

```bash
cd /home/jose/dev/vehiculos
```

## 2) Opcion recomendada (todo automatico)

Este comando crea/actualiza entorno, instala dependencias y ejecuta todo el flujo:

```bash
bash scripts/start_project.sh
```

## 3) Opcion manual paso a paso

1. Crear o actualizar entorno:

```bash
bash scripts/bootstrap_conda.sh
```

2. Activar entorno:

```bash
conda activate vehicular-ml
```

3. Ejecutar pipeline completo:

```bash
bash scripts/run_pipeline.sh
```

## 4) Ejecucion por comandos CLI

1. EDA:

```bash
python -m vehicular_ml.cli eda \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output-dir artifacts/eda
```

2. Entrenamiento:

```bash
python -m vehicular_ml.cli train \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output-dir artifacts/model \
  --test-size 0.2 \
  --random-state 42 \
  --n-estimators 300
```

3. Prediccion:

```bash
python -m vehicular_ml.cli predict \
  --model artifacts/model/model.joblib \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output artifacts/predicciones.csv
```

4. Tests:

```bash
python -m pytest -q
```

## 5) Salidas esperadas

Despues de correr correctamente, deben existir:

- `artifacts/eda/reporte_eda.md`
- `artifacts/model/model.joblib`
- `artifacts/model/metrics.json`
- `artifacts/model/test_predictions.csv`
- `artifacts/predicciones.csv`

## 6) Errores comunes

- `conda: command not found`:
  - Miniconda no esta instalado o no esta en PATH.
- `No module named vehicular_ml`:
  - Ejecuta `bash scripts/bootstrap_conda.sh` y activa `vehicular-ml`.
- `Faltan columnas requeridas`:
  - El dataset no tiene todas las columnas minimas definidas en `src/vehicular_ml/schema.py`.
