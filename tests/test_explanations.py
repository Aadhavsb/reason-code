from __future__ import annotations

import json
from pathlib import Path

from reason_code.explanations import (
    EvidenceKind,
    ExplanationClaim,
    ExplanationEvidence,
    FaithfulnessReasonCode,
    GeneratedExplanation,
    generate_explanation,
    grade_faithfulness,
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
        "Application requires human review. Debt-to-income ratio: 43%. "
        "Debt-to-income reason code."
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
        text="Employment history: 42%.",
        claims=(ExplanationClaim.model_validate(case["unfaithful_claim"]),),
    )

    verdict = grade_faithfulness(explanation, evidence)

    assert not verdict.faithful
    assert {reason.code for reason in verdict.reasons} == {
        FaithfulnessReasonCode.UNSUPPORTED_SUBJECT,
        FaithfulnessReasonCode.UNSUPPORTED_NUMBER,
    }
    assert [reason.code.value for reason in verdict.reasons] == case["expected_reason_codes"]


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
    assert verdict.reasons[0].code is FaithfulnessReasonCode.UNKNOWN_EVIDENCE_ID
