# Evaluacion del Notebook Primitivo y Como Funciona

## Fuentes revisadas

- `importante/primitivo.txt` (primer intento de notebook/codigo)
- `importante/02-09-2026_112116052_596DOCUMENTO.docx` (pautas de evaluacion parcial)
- `importante/ANÁLISIS DE MANTENIMIENTO VEHICULAR PARA LA PREDICCIÓN DE FALLAS DE MOTOR.docx` (informe academico)

## Como funciona el notebook primitivo (paso a paso)

1. Define un dataset pequeno en memoria (10 filas) con variables de operacion vehicular y `codigo_falla`.
2. Construye la variable objetivo `requiere_mantenimiento` con reglas de negocio:
   - kilometraje > 120000
   - temperatura_motor > 105
   - voltaje_bateria < 12
   - codigo de falla critico (`P0300`, `P0420`)
3. Separa `X` y `y`.
4. Crea pipeline de preprocesamiento con `ColumnTransformer`:
   - numericas: imputacion mediana + `StandardScaler`
   - categoricas: imputacion moda + `OneHotEncoder`
5. Entrena `RandomForestClassifier`.
6. Predice sobre test y reporta solo `accuracy`.

## Evaluacion tecnica contra las pautas

### Fortalezas

- Usa bibliotecas correctas del curso (`pandas`, `scikit-learn`).
- Incluye pipeline de preprocesamiento reproducible.
- Define objetivo con criterio de mantenimiento preventivo.
- Es un buen punto de partida didactico.

### Huecos detectados

- Hay leakage: el preprocesamiento se ajusta antes del `train_test_split`.
- Dataset extremadamente pequeno (10 registros), no representativo.
- Solo reporta `accuracy`; faltan `precision`, `recall`, `f1`, `roc_auc`.
- No guarda artefactos (`model.joblib`, `metrics.json`, predicciones).
- No hay CLI, ni estructura profesional (`src/`, `tests/`), ni pruebas automaticas.
- No hay manejo explicito de errores ni validacion robusta de esquema.

## Conclusiones

- El notebook primitivo cumple como demostracion inicial de concepto.
- Para nivel profesional/academico evaluable, requiere pipeline E2E reproducible, persistencia, pruebas y documentacion operativa.
- Esas brechas quedan cubiertas en la version actual del repositorio con CLI, artefactos, scripts Conda y tests.
