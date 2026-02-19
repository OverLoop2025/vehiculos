# Revision de Huecos Tecnicos del Proyecto

## Alcance de la revision

Se revisaron los siguientes grupos de archivos:

- Codigo fuente en `src/vehicular_ml/`
- Pruebas en `tests/`
- Scripts en `scripts/`
- Configuracion (`pyproject.toml`, `environment.yml`)
- Documentacion (`README.md`, carpeta `importante/`)
- Notebook `notebooks/vehicular_ml_workflow.ipynb`
- Artefactos generados en `artifacts/`

## Huecos tecnicos detectados (estado inicial)

1. Manejo de errores incompleto en CLI y prediccion.
2. Validacion de target binario poco estricta.
3. Riesgo de fallas silenciosas por tipos de columnas no normalizados.
4. Entrenamiento sin validacion explicita de configuracion (`test_size`, `n_estimators`).
5. Casos borde de distribucion de clases no reportados claramente.
6. Cobertura de tests incompleta para carga de datos y errores funcionales.
7. Falta de documento operativo especifico para correr todo desde cero.

## Correcciones aplicadas

1. Validacion y tipos de datos:
   - `src/vehicular_ml/data.py`
   - Se agrego normalizacion de tipos, validacion binaria de target y errores explicitos.
2. Robustez de entrenamiento:
   - `src/vehicular_ml/training.py`
   - Se agrego `TrainingError`, validacion de config, chequeo de distribucion de clases y mensajes claros.
3. Robustez de prediccion:
   - `src/vehicular_ml/predict.py`
   - Se agrego `PredictionError`, validacion de esquema y control de carga de modelo.
4. CLI mas profesional:
   - `src/vehicular_ml/cli.py`
   - Manejo centralizado de excepciones con codigos de salida consistentes.
5. EDA reproducible en entornos sin display:
   - `src/vehicular_ml/eda.py`
   - Backend `Agg` para generacion de figuras en local/CI.
6. Scripts de arranque:
   - `scripts/bootstrap_conda.sh`
   - `scripts/run_pipeline.sh`
   - `scripts/start_project.sh` (nuevo)
7. Pruebas funcionales ampliadas:
   - `tests/test_data.py`
   - `tests/test_training.py`
   - `tests/test_predict.py`
8. Documentacion operativa:
   - `importante/COMO_CORRER_TODO.md` (nuevo)

## Riesgos residuales (no bloqueantes)

1. El proyecto usa dataset sintetico; para despliegue real se requiere dataset productivo con control de sesgo.
2. No hay CI/CD remoto ni versionado de artefactos/experimentos (solo ejecucion local reproducible).
3. No hay monitoreo de drift ni recalibracion automatica del modelo.

## Estado final

El repo queda alineado a un flujo academico-profesional reproducible en local con:

- pipeline EDA + train + evaluacion + prediccion
- persistencia de artefactos
- CLI estable
- pruebas funcionales minimas automatizadas
- guia de ejecucion completa
