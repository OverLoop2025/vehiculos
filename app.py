"""Dashboard web para visualizar resultados del proyecto vehicular."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vehicular_ml.data import ensure_target_column, load_dataset
from vehicular_ml.eda import run_eda
from vehicular_ml.predict import predict_dataframe
from vehicular_ml.testing import run_pytest_with_summary
from vehicular_ml.training import TrainingConfig, train_and_save

st.set_page_config(page_title="Vehicular ML Dashboard", layout="wide")

DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "model" / "metrics.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "predicciones.csv"
DEFAULT_EDA_DIR = PROJECT_ROOT / "artifacts" / "eda"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "model.joblib"
DEFAULT_TEST_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "tests" / "summary.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "vehicular_mantenimiento_sample.csv"
EDA_FIGURE_NAMES = [
    "figura_1_distribucion_kilometraje.png",
    "figura_2_boxplot_temperatura.png",
    "figura_3_mapa_calor_correlacion.png",
    "figura_4_scatter_kilometraje_falla.png",
]


def _path_input(label: str, default_path: Path) -> Path:
    value = st.sidebar.text_input(label, str(default_path))
    return Path(value).expanduser()


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_tabular(path: str) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Formato no soportado: {suffix}")


def _status_line(container: Any, path: Path, label: str) -> None:
    if path.exists():
        container.success(f"{label}: OK")
    else:
        container.warning(f"{label}: No encontrado ({path})")


def _copy_if_needed(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _eda_outputs_exist(eda_dir: Path) -> bool:
    return all((eda_dir / file_name).exists() for file_name in EDA_FIGURE_NAMES)


def _artifacts_ready(
    metrics_path: Path,
    predictions_path: Path,
    eda_dir: Path,
    model_path: Path,
) -> bool:
    return (
        metrics_path.exists()
        and predictions_path.exists()
        and model_path.exists()
        and _eda_outputs_exist(eda_dir)
    )


def generate_demo_artifacts(
    data_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    eda_dir: Path,
    model_path: Path,
    tests_summary_path: Path,
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"No existe dataset de demo: {data_path}")

    df = load_dataset(data_path)
    run_eda(df=df, output_dir=eda_dir)

    summary = train_and_save(
        df=df,
        output_dir=model_path.parent,
        config=TrainingConfig(test_size=0.2, random_state=42, n_estimators=300),
    )
    generated_model_path = Path(summary["artifacts"]["model"])
    generated_metrics_path = Path(summary["artifacts"]["metrics"])
    _copy_if_needed(generated_model_path, model_path)
    _copy_if_needed(generated_metrics_path, metrics_path)

    pred_df = predict_dataframe(model_path, df)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_path, index=False)

    run_pytest_with_summary(project_root=PROJECT_ROOT, output_dir=tests_summary_path.parent)


def _read_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(uploaded_file)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(uploaded_file)
    except (EmptyDataError, ParserError) as exc:
        raise ValueError(f"No se pudo leer el archivo cargado: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Formato invalido del archivo cargado: {exc}") from exc
    raise ValueError(f"Formato no soportado: {suffix}. Usa CSV o Excel.")


def render_metrics_section(metrics_path: Path) -> None:
    st.subheader("Metricas del modelo")
    if not metrics_path.exists():
        st.info("No se encontro metrics.json. Ejecuta entrenamiento o pipeline primero.")
        return

    try:
        payload = _read_json(str(metrics_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer metrics.json: {exc}")
        return

    metrics = payload.get("metrics", {})
    if not metrics:
        st.info("El archivo de metricas no contiene el bloque `metrics`.")
        return

    cols = st.columns(len(metrics))
    for idx, (name, value) in enumerate(metrics.items()):
        cols[idx].metric(name.upper(), f"{float(value):.4f}")

    with st.expander("Detalle completo (metrics.json)"):
        st.json(payload)


def render_model_context_section(metrics_path: Path) -> None:
    st.subheader("Contexto del modelo")
    if not metrics_path.exists():
        st.info("No hay contexto del modelo aun. Genera artefactos primero.")
        return

    try:
        payload = _read_json(str(metrics_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer contexto del modelo: {exc}")
        return

    overview = st.columns(3)
    overview[0].metric("Train rows", int(payload.get("train_rows", 0)))
    overview[1].metric("Test rows", int(payload.get("test_rows", 0)))
    overview[2].metric("Fecha artefactos", "Disponible" if payload.get("created_at_utc") else "N/D")

    created_at = payload.get("created_at_utc")
    if created_at:
        st.caption(f"Ultima generacion (UTC): {created_at}")

    config = payload.get("config", {})
    if config:
        with st.expander("Configuracion de entrenamiento"):
            st.json(config)

    confusion = payload.get("confusion_matrix")
    if isinstance(confusion, dict) and isinstance(confusion.get("matrix"), list):
        st.write("Matriz de confusion")
        labels = confusion.get("labels", [0, 1])
        matrix_df = pd.DataFrame(
            confusion["matrix"],
            index=[f"Real {label}" for label in labels],
            columns=[f"Pred {label}" for label in labels],
        )
        st.dataframe(matrix_df, width="content")

        c_cols = st.columns(4)
        c_cols[0].metric("TP", int(confusion.get("tp", 0)))
        c_cols[1].metric("TN", int(confusion.get("tn", 0)))
        c_cols[2].metric("FP", int(confusion.get("fp", 0)))
        c_cols[3].metric("FN", int(confusion.get("fn", 0)))

    importances = payload.get("feature_importances_top")
    if isinstance(importances, list) and importances:
        st.write("Importancia de variables (Top)")
        importance_df = pd.DataFrame(importances)
        st.dataframe(importance_df, width="stretch")
        if {"feature", "importance"}.issubset(importance_df.columns):
            chart_df = importance_df.set_index("feature")[["importance"]]
            st.bar_chart(chart_df)


def render_eda_section(eda_dir: Path) -> None:
    st.subheader("Graficos EDA")
    if not eda_dir.exists():
        st.info("No se encontro la carpeta EDA. Ejecuta el comando de EDA primero.")
        return

    cols = st.columns(2)
    for idx, file_name in enumerate(EDA_FIGURE_NAMES):
        path = eda_dir / file_name
        if path.exists():
            cols[idx % 2].image(str(path), caption=file_name, width="stretch")
        else:
            cols[idx % 2].warning(f"Falta {file_name}")


def render_predictions_section(predictions_path: Path) -> None:
    st.subheader("Predicciones por lote")
    if not predictions_path.exists():
        st.info("No se encontro `artifacts/predicciones.csv`.")
        return

    try:
        df = _read_tabular(str(predictions_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo cargar predicciones: {exc}")
        return

    st.write(f"Filas: {len(df)}")
    st.dataframe(df.head(200), width="stretch")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar predicciones",
        data=csv_bytes,
        file_name="predicciones.csv",
        mime="text/csv",
    )


def render_interactive_prediction(model_path: Path) -> None:
    st.subheader("Prediccion interactiva")
    st.caption("Sube un CSV o XLSX para inferencia con el modelo entrenado.")

    if not model_path.exists():
        st.info("No se encontro `model.joblib`. Entrena primero el modelo.")
        return

    uploaded = st.file_uploader("Archivo de entrada", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        return

    try:
        input_df = _read_uploaded_file(uploaded)
        normalized_df = ensure_target_column(input_df)
        pred_df = predict_dataframe(model_path, normalized_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo predecir con el archivo cargado: {exc}")
        return

    st.write(f"Filas predichas: {len(pred_df)}")
    st.dataframe(pred_df.head(200), width="stretch")
    st.download_button(
        label="Descargar predicciones del archivo cargado",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_upload.csv",
        mime="text/csv",
    )


def render_tests_section(tests_summary_path: Path) -> None:
    st.subheader("Calidad de software (tests)")

    if not tests_summary_path.exists():
        st.info("No hay resumen de tests. Ejecuta tests para generar contexto de calidad.")
        if st.button("Ejecutar tests ahora", key="run_tests_missing"):
            with st.spinner("Ejecutando tests y generando resumen..."):
                try:
                    run_pytest_with_summary(
                        project_root=PROJECT_ROOT,
                        output_dir=tests_summary_path.parent,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"No se pudieron ejecutar tests: {exc}")
                else:
                    st.success("Resumen de tests generado.")
                    st.rerun()
        return

    try:
        payload = _read_json(str(tests_summary_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer el resumen de tests: {exc}")
        return

    counts = payload.get("counts", {})
    status = str(payload.get("status", "unknown")).upper()
    generated_at = payload.get("generated_at_utc", "N/D")
    duration = float(payload.get("duration_seconds", 0.0))

    cols = st.columns(6)
    cols[0].metric("Estado", status)
    cols[1].metric("Tests", int(counts.get("tests", 0)))
    cols[2].metric("Passed", int(counts.get("passed", 0)))
    cols[3].metric("Failed", int(counts.get("failed", 0)))
    cols[4].metric("Errors", int(counts.get("errors", 0)))
    cols[5].metric("Skipped", int(counts.get("skipped", 0)))

    st.caption(f"Ultima ejecucion (UTC): {generated_at} | Duracion: {duration:.2f}s")

    if st.button("Re-ejecutar tests", key="run_tests_existing"):
        with st.spinner("Ejecutando tests y actualizando resumen..."):
            try:
                run_pytest_with_summary(
                    project_root=PROJECT_ROOT,
                    output_dir=tests_summary_path.parent,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudieron ejecutar tests: {exc}")
            else:
                st.success("Resumen de tests actualizado.")
                st.rerun()

    with st.expander("Detalle de resumen de tests"):
        st.json(payload)


def main() -> None:
    st.title("Vehicular ML Dashboard")
    st.caption("Visualizacion de metricas, EDA y predicciones del proyecto.")

    st.sidebar.header("Rutas de artefactos")
    data_path = _path_input("Dataset demo", DEFAULT_DATA_PATH)
    metrics_path = _path_input("metrics.json", DEFAULT_METRICS_PATH)
    predictions_path = _path_input("predicciones.csv", DEFAULT_PREDICTIONS_PATH)
    eda_dir = _path_input("Directorio EDA", DEFAULT_EDA_DIR)
    model_path = _path_input("model.joblib", DEFAULT_MODEL_PATH)
    tests_summary_path = _path_input("tests summary", DEFAULT_TEST_SUMMARY_PATH)

    artifacts_ready = _artifacts_ready(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        eda_dir=eda_dir,
        model_path=model_path,
    )

    if not artifacts_ready:
        st.sidebar.markdown("---")
        st.sidebar.info(
            "No hay artefactos generados. Usa el boton para crear una demo automaticamente."
        )
        if st.sidebar.button("Generar artefactos demo"):
            with st.spinner("Generando artefactos (EDA + modelo + predicciones + tests)..."):
                try:
                    generate_demo_artifacts(
                        data_path=data_path,
                        metrics_path=metrics_path,
                        predictions_path=predictions_path,
                        eda_dir=eda_dir,
                        model_path=model_path,
                        tests_summary_path=tests_summary_path,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"No se pudo generar la demo: {exc}")
                else:
                    st.sidebar.success("Artefactos demo generados.")
                    st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.write("Estado de archivos")
    _status_line(st.sidebar, metrics_path, "Metricas")
    _status_line(st.sidebar, predictions_path, "Predicciones")
    _status_line(st.sidebar, eda_dir, "EDA")
    _status_line(st.sidebar, model_path, "Modelo")
    _status_line(st.sidebar, tests_summary_path, "Tests")

    render_metrics_section(metrics_path)
    st.markdown("---")
    render_model_context_section(metrics_path)
    st.markdown("---")
    render_eda_section(eda_dir)
    st.markdown("---")
    render_predictions_section(predictions_path)
    st.markdown("---")
    render_tests_section(tests_summary_path)
    st.markdown("---")
    render_interactive_prediction(model_path)


if __name__ == "__main__":
    main()
