from __future__ import annotations

from pathlib import Path

from vehicular_ml.testing import _parse_junit_xml


def test_parse_junit_xml_returns_counts(tmp_path: Path) -> None:
    junit_path = tmp_path / "junit.xml"
    junit_path.write_text(
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<testsuite name=\"pytest\" tests=\"5\" failures=\"1\" errors=\"0\" skipped=\"1\" time=\"0.75\">
  <testcase classname=\"tests.test_a\" name=\"ok\" time=\"0.1\"/>
</testsuite>
""",
        encoding="utf-8",
    )

    parsed = _parse_junit_xml(junit_path)

    assert parsed["tests"] == 5
    assert parsed["failures"] == 1
    assert parsed["errors"] == 0
    assert parsed["skipped"] == 1
    assert parsed["time"] == 0.75


def test_parse_junit_xml_handles_missing_file(tmp_path: Path) -> None:
    parsed = _parse_junit_xml(tmp_path / "missing.xml")

    assert parsed["tests"] == 0
    assert parsed["failures"] == 0
    assert parsed["errors"] == 0
    assert parsed["skipped"] == 0
