# Prediccion de Fallas de Motor y Mantenimiento Vehicular

Proyecto de Machine Learning para entrenar y usar un modelo de clasificacion sobre mantenimiento vehicular.

## Que incluye este proyecto

- Carga y validacion de dataset vehicular.
- EDA con generacion de reporte y figuras.
- Preprocesamiento con `ColumnTransformer`:
  - imputacion numerica y categorica
  - `OneHotEncoder`
  - `StandardScaler`
- Entrenamiento de modelo con `RandomForestClassifier`.
- Evaluacion del modelo y guardado de metricas.
- Prediccion por lote desde CLI.
- Tests funcionales con `pytest`.

## Estructura principal

```text
vehiculos/
  app.py
  artifacts/
  data/raw/
  notebooks/
  scripts/
  src/vehicular_ml/
  tests/
  environment.yml
  pyproject.toml
  README.md
```

## Requisitos

- Miniconda o Anaconda
- Python 3.11

## Instalacion y arranque rapido

Desde la raiz del proyecto:

```bash
cd /home/jose/dev/vehiculos
bash scripts/bootstrap_conda.sh
conda activate vehicular-ml
```

Para ejecutar todo el flujo completo:

```bash
bash scripts/run_pipeline.sh
```

Para abrir el dashboard web con resultados:

```bash
bash scripts/run_dashboard.sh
```

Para despliegue en Streamlit Community Cloud usa:

- `Main file path`: `deploy/app.py`
- Dependencias: `deploy/requirements.txt`

## Comandos principales

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

3. Prediccion

```bash
python -m vehicular_ml.cli predict \
  --model artifacts/model/model.joblib \
  --data data/raw/vehicular_mantenimiento_sample.csv \
  --output artifacts/predicciones.csv
```

4. Tests

```bash
python -m pytest -q
```

5. Dashboard web (Streamlit)

```bash
python -m streamlit run app.py
```

## Salidas esperadas

- `artifacts/eda/reporte_eda.md`
- `artifacts/eda/*.png`
- `artifacts/model/model.joblib`
- `artifacts/model/metrics.json`
- `artifacts/model/test_predictions.csv`
- `artifacts/predicciones.csv`
