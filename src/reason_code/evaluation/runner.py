"""Fail-closed runner for the repository's frozen synthetic explanation fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reason_code.explanations import (
    ExplanationClaim,
    ExplanationEvidence,
    GeneratedExplanation,
    generate_explanation,
    grade_faithfulness,
)

SYNTHETIC_STATUS = "SYNTHETIC SCAFFOLD — not reported gold-set results"
MANIFEST_STATUS = "SYNTHETIC SCAFFOLD — not a real metric"
FIXTURE_VERSION = "synthetic-scaffold-v1"
MANIFEST_VERSION = "deterministic-evaluation-manifest-v1"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE_PATH = (
    _REPOSITORY_ROOT / "tests" / "fixtures" / "explanation_eval_synthetic_scaffold.json"
)
DEFAULT_MANIFEST_PATH = (
    _REPOSITORY_ROOT / "tests" / "fixtures" / "deterministic_evaluation_manifest.json"
)


class EvaluationError(ValueError):
    """Raised when frozen evaluation inputs or expectations cannot be trusted."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid {label}") from error
    if not isinstance(raw, dict):
        raise EvaluationError(f"invalid {label}")
    return raw


def _require_exact_keys(raw: dict[str, Any], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise EvaluationError(f"invalid {label} schema")


def _validate_evidence(raw: Any, case_id: str) -> ExplanationEvidence:
    if not isinstance(raw, dict):
        raise EvaluationError("invalid fixture evidence schema")
    _require_exact_keys(raw, {"case_id", "items"}, "fixture evidence")
    if raw["case_id"] != case_id or not isinstance(raw["items"], list):
        raise EvaluationError("invalid fixture evidence schema")
    for item in raw["items"]:
        if not isinstance(item, dict):
            raise EvaluationError("invalid fixture evidence schema")
        allowed = {"evidence_id", "kind", "label", "value", "detail"}
        if not {"evidence_id", "kind", "label"}.issubset(item) or not set(item) <= allowed:
            raise EvaluationError("invalid fixture evidence schema")
    try:
        return ExplanationEvidence.model_validate(raw)
    except ValidationError as error:
        raise EvaluationError("invalid fixture evidence schema") from error


def _candidate_for_case(raw: dict[str, Any], evidence: ExplanationEvidence) -> GeneratedExplanation:
    if raw["expected_faithful"] is True:
        return generate_explanation(evidence)

    claim_raw = raw.get("unfaithful_claim")
    if not isinstance(claim_raw, dict):
        raise EvaluationError("invalid fixture case schema")
    _require_exact_keys(
        claim_raw,
        {"evidence_id", "kind", "subject", "statement"},
        "fixture unfaithful_claim",
    )
    try:
        claim = ExplanationClaim.model_validate(claim_raw)
        return GeneratedExplanation(
            case_id=evidence.case_id,
            text=claim.statement,
            claims=(claim,),
        )
    except ValidationError as error:
        raise EvaluationError("invalid fixture case schema") from error


def _validate_fixture(raw: dict[str, Any]) -> list[dict[str, Any]]:
    _require_exact_keys(raw, {"fixture_version", "status", "description", "cases"}, "fixture")
    if raw["fixture_version"] != FIXTURE_VERSION or raw["status"] != SYNTHETIC_STATUS:
        raise EvaluationError("invalid fixture metadata")
    if not isinstance(raw["description"], str) or not isinstance(raw["cases"], list):
        raise EvaluationError("invalid fixture schema")

    case_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for raw_case in raw["cases"]:
        if not isinstance(raw_case, dict):
            raise EvaluationError("invalid fixture case schema")
        expected_keys = {"case_id", "evidence", "expected_faithful"}
        if raw_case.get("expected_faithful") is False:
            expected_keys |= {"unfaithful_claim", "expected_reason_codes"}
        _require_exact_keys(raw_case, expected_keys, "fixture case")
        case_id = raw_case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise EvaluationError("invalid fixture case schema")
        if type(raw_case["expected_faithful"]) is not bool:
            raise EvaluationError("invalid fixture case schema")
        if raw_case["expected_faithful"] is False:
            reason_codes = raw_case["expected_reason_codes"]
            if not isinstance(reason_codes, list) or not all(
                isinstance(code, str) for code in reason_codes
            ):
                raise EvaluationError("invalid fixture case schema")
        evidence = _validate_evidence(raw_case["evidence"], case_id)
        case_ids.add(case_id)
        cases.append({"raw": raw_case, "evidence": evidence})
    if not cases:
        raise EvaluationError("invalid fixture schema")
    return cases


def _validate_manifest(raw: dict[str, Any], fixture_path: Path) -> dict[str, int]:
    _require_exact_keys(
        raw,
        {"manifest_version", "status", "fixture", "fixture_sha256", "expected_summary"},
        "manifest",
    )
    if raw["manifest_version"] != MANIFEST_VERSION or raw["status"] != MANIFEST_STATUS:
        raise EvaluationError("invalid manifest metadata")
    if raw["fixture"] != fixture_path.name or not isinstance(raw["fixture_sha256"], str):
        raise EvaluationError("invalid manifest schema")
    fixture_content = fixture_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    fixture_hash = hashlib.sha256(fixture_content.encode("utf-8")).hexdigest()
    if raw["fixture_sha256"] != fixture_hash:
        raise EvaluationError("fixture checksum mismatch")
    summary = raw["expected_summary"]
    if not isinstance(summary, dict):
        raise EvaluationError("invalid manifest schema")
    _require_exact_keys(
        summary,
        {"case_count", "faithful_count", "unfaithful_count"},
        "manifest summary",
    )
    if any(type(value) is not int or value < 0 for value in summary.values()):
        raise EvaluationError("invalid manifest schema")
    return summary


def run_deterministic_evaluation(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Evaluate the frozen synthetic scaffold and return a stable JSON-ready report."""
    fixture_raw = _load_json_object(fixture_path, "fixture")
    manifest_raw = _load_json_object(manifest_path, "manifest")
    expected_summary = _validate_manifest(manifest_raw, fixture_path)
    cases = _validate_fixture(fixture_raw)

    results: list[dict[str, Any]] = []
    for case in cases:
        raw_case = case["raw"]
        evidence = case["evidence"]
        candidate = _candidate_for_case(raw_case, evidence)
        verdict = grade_faithfulness(candidate, evidence)
        reason_codes = [reason.code.value for reason in verdict.reasons]
        expected_reason_codes = raw_case.get("expected_reason_codes", [])
        if (
            verdict.faithful is not raw_case["expected_faithful"]
            or reason_codes != expected_reason_codes
        ):
            raise EvaluationError(f"fixture expectation mismatch for {evidence.case_id}")
        results.append(
            {
                "case_id": evidence.case_id,
                "faithful": verdict.faithful,
                "reason_codes": reason_codes,
            }
        )

    actual_summary = {
        "case_count": len(results),
        "faithful_count": sum(result["faithful"] for result in results),
        "unfaithful_count": sum(not result["faithful"] for result in results),
    }
    if actual_summary != expected_summary:
        raise EvaluationError("baseline summary mismatch")
    return {
        "baseline": "synthetic scaffold — not a real metric",
        "cases": results,
        "fixture_version": FIXTURE_VERSION,
        "mode": "deterministic",
        "status": "pass",
        "summary": actual_summary,
    }
