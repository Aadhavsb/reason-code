from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reason_code.evaluation import EvaluationError, run_deterministic_evaluation
from reason_code.evaluation.runner import DEFAULT_FIXTURE_PATH, DEFAULT_MANIFEST_PATH


def test_frozen_scaffold_matches_the_synthetic_baseline() -> None:
    report = run_deterministic_evaluation()

    assert report == {
        "baseline": "synthetic scaffold — not a real metric",
        "cases": [
            {
                "case_id": "synthetic-faithful-001",
                "faithful": True,
                "reason_codes": [],
            },
            {
                "case_id": "synthetic-unfaithful-001",
                "faithful": False,
                "reason_codes": [
                    "unsupported_subject",
                    "unsupported_statement",
                    "unsupported_number",
                ],
            },
        ],
        "fixture_version": "synthetic-scaffold-v1",
        "mode": "deterministic",
        "status": "pass",
        "summary": {"case_count": 2, "faithful_count": 1, "unfaithful_count": 1},
    }


def test_module_cli_emits_stable_json() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reason_code.evaluation.run", "--mode", "deterministic"],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        },
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        '{"baseline":"synthetic scaffold — not a real metric","cases":['
        '{"case_id":"synthetic-faithful-001","faithful":true,"reason_codes":[]},'
        '{"case_id":"synthetic-unfaithful-001","faithful":false,"reason_codes":['
        '"unsupported_subject","unsupported_statement","unsupported_number"]}],'
        '"fixture_version":"synthetic-scaffold-v1","mode":"deterministic",'
        '"status":"pass","summary":{"case_count":2,"faithful_count":1,'
        '"unfaithful_count":1}}\n'
    )


def test_unknown_fixture_fields_fail_closed(tmp_path: Path) -> None:
    fixture = json.loads(DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["unexpected"] = "rejected"
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["fixture"] = fixture_path.name
    manifest["fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EvaluationError, match="invalid fixture schema"):
        run_deterministic_evaluation(fixture_path, manifest_path)


def test_manifest_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(DEFAULT_FIXTURE_PATH.read_bytes())
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["fixture"] = fixture_path.name
    manifest["fixture_sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EvaluationError, match="fixture checksum mismatch"):
        run_deterministic_evaluation(fixture_path, manifest_path)
