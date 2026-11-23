from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from reason_code.explanations import (
    EvidenceItem,
    EvidenceKind,
    ExplanationClaim,
    ExplanationEvidence,
    FaithfulnessReason,
    FaithfulnessReasonCode,
    FaithfulnessVerdict,
    GeneratedExplanation,
    generate_explanation,
    grade_faithfulness,
    grade_prose_evidence,
    render_evidence_item,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "explanation_eval_synthetic_scaffold.json"


def _fixture_cases() -> dict[str, dict[str, object]]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["status"] == "SYNTHETIC SCAFFOLD — not reported gold-set results"
    return {case["case_id"]: case for case in fixture["cases"]}


def test_generator_is_deterministic_and_evidence_bounded() -> None:
    case = _fixture_cases()["synthetic-faithful-001"]
    evidence = ExplanationEvidence.model_validate(case["evidence"])

    first = generate_explanation(evidence)
    second = generate_explanation(evidence)

    assert first == second
    assert [claim.evidence_id for claim in first.claims] == [
        "decision",
        "debt_to_income",
        "reason_dti",
    ]
    assert first.text == (
        "Application requires human review. Debt-to-income ratio: 43%. Debt-to-income reason code."
    )


def test_generated_explanation_passes_programmatic_faithfulness_check() -> None:
    case = _fixture_cases()["synthetic-faithful-001"]
    evidence = ExplanationEvidence.model_validate(case["evidence"])

    verdict = grade_faithfulness(generate_explanation(evidence), evidence)

    assert verdict.faithful is case["expected_faithful"]
    assert verdict.reasons == ()


def test_grader_rejects_unsupported_factor_and_number() -> None:
    case = _fixture_cases()["synthetic-unfaithful-001"]
    evidence = ExplanationEvidence.model_validate(case["evidence"])
    explanation = GeneratedExplanation(
        case_id="synthetic-unfaithful-001",
        text="Employment history: 42%.",
        claims=(ExplanationClaim.model_validate(case["unfaithful_claim"]),),
    )

    verdict = grade_faithfulness(explanation, evidence)

    assert not verdict.faithful
    assert {reason.code for reason in verdict.reasons} == {
        FaithfulnessReasonCode.UNSUPPORTED_SUBJECT,
        FaithfulnessReasonCode.UNSUPPORTED_STATEMENT,
        FaithfulnessReasonCode.UNSUPPORTED_NUMBER,
    }
    assert {reason.code.value for reason in verdict.reasons} == set(case["expected_reason_codes"])


def test_grader_rejects_factor_appended_to_supported_statement() -> None:
    evidence = ExplanationEvidence.model_validate(
        {
            "case_id": "statement-injection",
            "items": [
                {
                    "evidence_id": "debt_to_income",
                    "kind": "factor",
                    "label": "Debt-to-income ratio",
                    "value": "43%",
                }
            ],
        }
    )
    explanation = GeneratedExplanation(
        case_id="statement-injection",
        text="Debt-to-income ratio: 43%. Employment history was also considered.",
        claims=(
            ExplanationClaim(
                evidence_id="debt_to_income",
                kind=EvidenceKind.FACTOR,
                subject="Debt-to-income ratio",
                statement=("Debt-to-income ratio: 43%. Employment history was also considered."),
            ),
        ),
    )

    verdict = grade_faithfulness(explanation, evidence)

    assert not verdict.faithful
    assert [reason.code for reason in verdict.reasons] == [
        FaithfulnessReasonCode.UNSUPPORTED_STATEMENT
    ]


def test_grader_rejects_tampered_text_and_case_binding() -> None:
    evidence = ExplanationEvidence.model_validate(
        {
            "case_id": "bound-case",
            "items": [
                {
                    "evidence_id": "decision",
                    "kind": "decision",
                    "label": "Human review",
                }
            ],
        }
    )
    explanation = GeneratedExplanation(
        case_id="other-case",
        text="Denied because of a 710 credit score.",
        claims=(
            ExplanationClaim(
                evidence_id="decision",
                kind=EvidenceKind.DECISION,
                subject="Human review",
                statement="Human review.",
            ),
        ),
    )

    verdict = grade_faithfulness(explanation, evidence)

    assert not verdict.faithful
    assert {reason.code for reason in verdict.reasons} == {
        FaithfulnessReasonCode.CASE_ID_MISMATCH,
        FaithfulnessReasonCode.TEXT_TRACE_MISMATCH,
    }


def test_grader_rejects_missing_evidence_claim() -> None:
    evidence = ExplanationEvidence.model_validate(
        {
            "case_id": "coverage-case",
            "items": [
                {"evidence_id": "decision", "kind": "decision", "label": "Human review"},
                {
                    "evidence_id": "reason_code",
                    "kind": "reason_code",
                    "label": "Debt-to-income reason code",
                },
            ],
        }
    )
    explanation = GeneratedExplanation(
        case_id="coverage-case",
        text="Human review.",
        claims=(
            ExplanationClaim(
                evidence_id="decision",
                kind=EvidenceKind.DECISION,
                subject="Human review",
                statement="Human review.",
            ),
        ),
    )

    verdict = grade_faithfulness(explanation, evidence)

    assert not verdict.faithful
    assert [reason.code for reason in verdict.reasons] == [
        FaithfulnessReasonCode.CLAIM_COVERAGE_MISMATCH
    ]


def test_grader_rejects_unknown_evidence_id() -> None:
    evidence = ExplanationEvidence.model_validate(
        {
            "case_id": "unknown-evidence",
            "items": [
                {
                    "evidence_id": "decision",
                    "kind": "decision",
                    "label": "Human review",
                }
            ],
        }
    )
    explanation = GeneratedExplanation(
        case_id="unknown-evidence",
        text="External score: 710.",
        claims=(
            ExplanationClaim(
                evidence_id="external_score",
                kind=EvidenceKind.METRIC,
                subject="External score",
                statement="External score: 710.",
            ),
        ),
    )

    verdict = grade_faithfulness(explanation, evidence)

    assert not verdict.faithful
    assert {reason.code for reason in verdict.reasons} == {
        FaithfulnessReasonCode.UNKNOWN_EVIDENCE_ID,
        FaithfulnessReasonCode.CLAIM_COVERAGE_MISMATCH,
    }


@pytest.mark.parametrize("case_id", ["", "   "])
def test_explanation_models_reject_empty_case_ids(case_id: str) -> None:
    with pytest.raises(ValidationError):
        ExplanationEvidence(
            case_id=case_id,
            items=({"evidence_id": "decision", "kind": "decision", "label": "Human review"},),
        )


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, "   "])
def test_evidence_rejects_boolean_nonfinite_and_empty_values(value: object) -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(evidence_id="metric", kind="metric", label="Metric", value=value)


def test_explanation_models_forbid_extra_fields_and_canonicalize_rendering() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(evidence_id="metric", kind="metric", label="Metric", extra="no")
    assert (
        render_evidence_item(
            EvidenceItem(evidence_id="metric", kind="metric", label="Metric", value=1.0)
        )
        == "Metric: 1."
    )
    assert (
        render_evidence_item(
            EvidenceItem(evidence_id="metric", kind="metric", label="Metric", detail="Details")
        )
        == "Metric: Details."
    )


def test_faithfulness_verdict_requires_boolean_to_match_reasons() -> None:
    reason = FaithfulnessReason(code="unsupported_statement", message="Unsupported.")
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(faithful=True, reasons=(reason,))
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(faithful=False, reasons=())


def test_prose_evidence_grader_is_distinct_from_strict_structural_gate() -> None:
    evidence = ExplanationEvidence(
        case_id="prose",
        items=({"evidence_id": "decision", "kind": "decision", "label": "Human review"},),
    )
    explanation = GeneratedExplanation(
        case_id="prose",
        text="Human review requires a person to review this application.",
        claims=(
            ExplanationClaim(
                evidence_id="decision",
                kind="decision",
                subject="Human review",
                statement="Human review requires a person to review this application.",
            ),
        ),
    )
    assert grade_prose_evidence(explanation, evidence).faithful is True
    assert grade_faithfulness(explanation, evidence).faithful is False
