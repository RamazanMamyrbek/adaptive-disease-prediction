import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = sorted((ROOT / "reports").glob("*.json"))


def referenced_hashes(value):
    if isinstance(value, list):
        for item in value:
            yield from referenced_hashes(item)
    elif isinstance(value, dict):
        for key, expected in value.items():
            if key == "artifact_migration":
                continue
            if key == "input_sha256":
                yield from expected.items()
            elif key == "sha256":
                path = value.get("path", value.get("local_path"))
                if path:
                    yield path, expected
            elif key.endswith("_sha256"):
                path = value.get(key.removesuffix("_sha256") + "_path")
                if path:
                    yield path, expected
            yield from referenced_hashes(expected)


@pytest.mark.parametrize("report", REPORTS, ids=lambda path: path.stem)
def test_report_references_match_available_artifacts(report: Path) -> None:
    document = json.loads(report.read_text(encoding="utf-8"))
    for name, expected in referenced_hashes(document):
        relative = Path(name)
        assert not relative.is_absolute(), f"Non-portable reference in {report.name}: {name}"
        path = ROOT / relative
        if not path.exists() and relative.parts[0] in {"data", "models"}:
            continue
        assert path.is_file(), f"Missing reference in {report.name}: {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Checksum mismatch in {report.name}: {name}"
