"""Ejecucion de tests y generacion de resumen persistente."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _parse_junit_xml(junit_path: Path) -> dict[str, Any]:
    if not junit_path.exists():
        return {
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "time": 0.0,
        }

    try:
        root = ET.fromstring(junit_path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        return {
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "time": 0.0,
            "parse_error": str(exc),
        }

    suites: list[ET.Element] = []
    if root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    elif root.tag == "testsuite":
        suites = [root]

    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    total_time = sum(float(suite.attrib.get("time", "0.0")) for suite in suites)

    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "time": total_time,
    }


def _tail(text: str, lines: int = 30) -> str:
    content = text.strip()
    if not content:
        return ""
    return "\n".join(content.splitlines()[-lines:])


def run_pytest_with_summary(
    project_root: str | Path,
    output_dir: str | Path,
    extra_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    junit_path = output_dir / "junit.xml"
    summary_path = output_dir / "summary.json"

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--junitxml={junit_path}",
    ]
    if extra_args:
        command.extend(extra_args)

    result = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    parsed = _parse_junit_xml(junit_path)
    tests = int(parsed["tests"])
    failures = int(parsed["failures"])
    errors = int(parsed["errors"])
    skipped = int(parsed["skipped"])
    passed = max(tests - failures - errors - skipped, 0)

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if result.returncode == 0 else "failed",
        "exit_code": int(result.returncode),
        "duration_seconds": float(parsed["time"]),
        "counts": {
            "tests": tests,
            "passed": passed,
            "failed": failures,
            "errors": errors,
            "skipped": skipped,
        },
        "command": " ".join(command),
        "artifacts": {
            "junit_xml": str(junit_path),
            "summary": str(summary_path),
        },
    }

    parse_error = parsed.get("parse_error")
    if parse_error:
        summary["parse_error"] = parse_error

    stdout_tail = _tail(result.stdout)
    stderr_tail = _tail(result.stderr)
    if stdout_tail:
        summary["stdout_tail"] = stdout_tail
    if stderr_tail:
        summary["stderr_tail"] = stderr_tail

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
