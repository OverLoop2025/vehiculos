"""Dashboard web para visualizar resultados del proyecto vehicular."""

from __future__ import annotations

import json
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

from vehicular_ml.data import ensure_target_column
from vehicular_ml.predict import predict_dataframe

st.set_page_config(page_title="Vehicular ML Dashboard", layout="wide")

DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "model" / "metrics.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "predicciones.csv"
DEFAULT_EDA_DIR = PROJECT_ROOT / "artifacts" / "eda"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "model.joblib"


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


def render_eda_section(eda_dir: Path) -> None:
    st.subheader("Graficos EDA")
    figure_names = [
        "figura_1_distribucion_kilometraje.png",
        "figura_2_boxplot_temperatura.png",
        "figura_3_mapa_calor_correlacion.png",
        "figura_4_scatter_kilometraje_falla.png",
    ]

    if not eda_dir.exists():
        st.info("No se encontro la carpeta EDA. Ejecuta el comando de EDA primero.")
        return

    cols = st.columns(2)
    for idx, file_name in enumerate(figure_names):
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


def main() -> None:
    st.title("Vehicular ML Dashboard")
    st.caption("Visualizacion de metricas, EDA y predicciones del proyecto.")

    st.sidebar.header("Rutas de artefactos")
    metrics_path = _path_input("metrics.json", DEFAULT_METRICS_PATH)
    predictions_path = _path_input("predicciones.csv", DEFAULT_PREDICTIONS_PATH)
    eda_dir = _path_input("Directorio EDA", DEFAULT_EDA_DIR)
    model_path = _path_input("model.joblib", DEFAULT_MODEL_PATH)

    st.sidebar.markdown("---")
    st.sidebar.write("Estado de archivos")
    _status_line(st.sidebar, metrics_path, "Metricas")
    _status_line(st.sidebar, predictions_path, "Predicciones")
    _status_line(st.sidebar, eda_dir, "EDA")
    _status_line(st.sidebar, model_path, "Modelo")

    render_metrics_section(metrics_path)
    st.markdown("---")
    render_eda_section(eda_dir)
    st.markdown("---")
    render_predictions_section(predictions_path)
    st.markdown("---")
    render_interactive_prediction(model_path)


if __name__ == "__main__":
    main()
