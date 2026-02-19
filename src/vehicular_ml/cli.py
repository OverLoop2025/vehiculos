"""CLI del proyecto vehicular de mantenimiento predictivo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vehicular_ml.data import DatasetValidationError, load_dataset
from vehicular_ml.eda import run_eda
from vehicular_ml.predict import PredictionError, predict_dataframe
from vehicular_ml.training import TrainingConfig, TrainingError, train_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vehicular-ml",
        description="EDA, entrenamiento y prediccion para fallas de motor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda_parser = subparsers.add_parser("eda", help="Genera analisis exploratorio de datos")
    eda_parser.add_argument("--data", required=True, help="Ruta a CSV/XLSX")
    eda_parser.add_argument("--output-dir", default="artifacts/eda", help="Carpeta salida")

    train_parser = subparsers.add_parser("train", help="Entrena modelo y guarda artefactos")
    train_parser.add_argument("--data", required=True, help="Ruta a CSV/XLSX")
    train_parser.add_argument(
        "--output-dir", default="artifacts/model", help="Carpeta de artefactos"
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--n-estimators", type=int, default=300)

    predict_parser = subparsers.add_parser("predict", help="Predice sobre nuevos registros")
    predict_parser.add_argument("--model", required=True, help="Ruta al model.joblib")
    predict_parser.add_argument("--data", required=True, help="Ruta a CSV/XLSX")
    predict_parser.add_argument(
        "--output",
        default="artifacts/predicciones.csv",
        help="Archivo CSV de salida",
    )

    return parser


def cmd_eda(args: argparse.Namespace) -> None:
    df = load_dataset(args.data)
    outputs = run_eda(df=df, output_dir=args.output_dir)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


def cmd_train(args: argparse.Namespace) -> None:
    df = load_dataset(args.data)
    config = TrainingConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    summary = train_and_save(df=df, output_dir=args.output_dir, config=config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_predict(args: argparse.Namespace) -> None:
    df = load_dataset(args.data)
    predictions = predict_dataframe(args.model, df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(
        json.dumps(
            {
                "rows": len(predictions),
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "eda":
            cmd_eda(args)
        elif args.command == "train":
            cmd_train(args)
        elif args.command == "predict":
            cmd_predict(args)
        else:
            parser.error(f"Comando no reconocido: {args.command}")
        return 0
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except (DatasetValidationError, TrainingError, PredictionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
