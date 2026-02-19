# Prompt Maestro Para Codex (Recomendado: Codex CLI)

## Plataforma elegida

Usa **Codex CLI** para este proyecto porque requiere:

- crear/editar muchos archivos
- instalar dependencias y administrar entorno Conda
- ejecutar scripts y pruebas automatizadas
- iterar rapido con trazabilidad tecnica

## Prompt listo para usar

```text
Actua como ingeniero senior de Machine Learning y MLOps.

Contexto:
Estoy en un proyecto de mantenimiento vehicular para prediccion de fallas de motor.
Necesito una solucion profesional, reproducible y ejecutable en local.

Objetivo:
1) Te deje archivos de guia importantes en la carpeta importante, revisalos que son nuestra metrica en general, ademas del desarrollo de un notebook primitivo, es decir como un primer intento simple, creame un archivo .md en donde lo evalues y me digas como funciona
2) Revisar todos los archivos del proyecto y detectar huecos tecnicos.
3) Dejar pipeline completo de EDA + preprocesamiento + entrenamiento + evaluacion + prediccion.
4) Configurar y usar Miniconda correctamente (crear/actualizar entorno, instalar dependencias y dejar scripts de arranque).
5) Agregar pruebas automaticas y ejecutarlas.
6) Entregar documentacion clara de uso y comandos finales para produccion academica.

Requisitos de implementacion:
- Python 3.11
- Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib, Pytest
- Pipeline con ColumnTransformer (imputacion, OneHotEncoder, StandardScaler)
- Modelo base RandomForestClassifier con metricas: accuracy, precision, recall, f1 y roc_auc cuando aplique
- Persistencia de artefactos (model.joblib, metrics.json, predicciones)
- CLI con comandos: eda, train, predict
- Tests funcionales minimos: carga/esquema, entrenamiento, prediccion

Criterios de calidad:
- Codigo limpio, modular y tipado
- Manejo de errores explicito
- Sin hardcode innecesario
- Estructura profesional con src/ y tests/
- README con quickstart y flujo completo

Modo de trabajo:
- Ejecuta cambios directamente en el repo.
- Antes de editar, explica en 1-2 lineas que haras.
- Al final, muestra resumen de cambios por archivo, resultados de tests y siguientes pasos.
- Si falta informacion bloqueante, asume defaults razonables y documentalos.
```
