#!/usr/bin/env python
"""Ejecuta pytest y guarda resumen estructurado para dashboard/reportes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vehicular_ml.testing import run_pytest_with_summary


def main() -> int:
    summary = run_pytest_with_summary(
        project_root=PROJECT_ROOT,
        output_dir=PROJECT_ROOT / "artifacts" / "tests",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
