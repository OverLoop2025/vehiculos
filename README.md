# Prediccion de Fallas de Motor y Mantenimiento Vehicular

Solucion de Machine Learning y MLOps local para el caso academico:
`Analisis de mantenimiento vehicular para la prediccion de fallas de motor`.

## Alcance

El proyecto implementa un flujo completo reproducible:

- EDA con reporte y figuras (`matplotlib` / `seaborn`)
- Preprocesamiento con `ColumnTransformer`:
  - imputacion numerica y categorica
  - `OneHotEncoder`
  - `StandardScaler`
- Entrenamiento con `RandomForestClassifier`
- Evaluacion con `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (cuando aplica)
- Persistencia de artefactos:
  - `model.joblib`
  - `metrics.json`
  - predicciones CSV
- CLI con comandos `eda`, `train`, `predict`
- Tests funcionales con `pytest`

## Estructura

```text
vehiculos/
  artifacts/
  data/raw/
  importante/
  notebooks/
  scripts/
  src/vehicular_ml/
  tests/
  environment.yml
  pyproject.toml
  README.md
```

## Requisitos

- Miniconda/Anaconda
- Python 3.11 (gestionado por `environment.yml`)

Dependencias principales:
`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `pytest`.

## Quickstart con Miniconda

Desde la raiz del proyecto:

```bash
cd /home/jose/dev/vehiculos
bash scripts/bootstrap_conda.sh
conda activate vehicular-ml
```

Ejecucion de todo en un solo comando:

```bash
bash scripts/start_project.sh
```

## CLI del proyecto

1. EDA

```bash
python -m vehicular_ml.cli eda \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output-dir artifacts/eda
```

2. Entrenamiento

```bash
python -m vehicular_ml.cli train \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output-dir artifacts/model \
  --test-size 0.2 \
  --random-state 42 \
  --n-estimators 300
```

3. Prediccion por lote

```bash
python -m vehicular_ml.cli predict \
  --model artifacts/model/model.joblib \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output artifacts/predicciones.csv
```

## Flujo completo reproducible

```bash
bash scripts/run_pipeline.sh
```

Este script ejecuta:

1. `scripts/generate_sample_data.py`
2. `vehicular_ml.cli eda`
3. `vehicular_ml.cli train`
4. `vehicular_ml.cli predict`
5. `python -m pytest -q`

## Dataset esperado

Columnas minimas obligatorias:

- `kilometraje`
- `temperatura_motor`
- `rpm`
- `horas_uso`
- `voltaje_bateria`
- `marca`
- `combustible`
- `tipo_aceite`
- `codigo_falla`

La columna `requiere_mantenimiento` es opcional. Si no existe, se genera por reglas de negocio.

## Artefactos esperados

- `artifacts/eda/reporte_eda.md`
- `artifacts/eda/figura_1_distribucion_kilometraje.png`
- `artifacts/eda/figura_2_boxplot_temperatura.png`
- `artifacts/eda/figura_3_mapa_calor_correlacion.png`
- `artifacts/eda/figura_4_scatter_kilometraje_falla.png`
- `artifacts/model/model.joblib`
- `artifacts/model/metrics.json`
- `artifacts/model/test_predictions.csv`
- `artifacts/predicciones.csv`

## Tests

```bash
python -m pytest -q
```

Cobertura funcional minima:

- carga de datos y esquema
- entrenamiento y artefactos
- prediccion y probabilidades

## Documentos de evaluacion

- Guia de ejecucion completa:
  - `importante/COMO_CORRER_TODO.md`
- Evaluacion del notebook primitivo y guias:
  - `importante/evaluacion_notebook_primitivo.md`
- Revision de huecos tecnicos del proyecto:
  - `importante/revision_huecos_tecnicos.md`
