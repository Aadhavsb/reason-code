"""Deterministic structural and prose/evidence checks for explanation claim traces."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .generator import render_evidence_item
from .models import (
    EvidenceItem,
    ExplanationEvidence,
    FaithfulnessReason,
    FaithfulnessReasonCode,
    FaithfulnessVerdict,
    GeneratedExplanation,
)

_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])-?\$?\d[\d,]*(?:\.\d+)?%?")


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _number_signature(token: str) -> tuple[Decimal, bool, bool] | None:
    is_percent = token.endswith("%")
    raw_value = token.removesuffix("%").replace(",", "")
    is_currency = "$" in raw_value
    raw_value = raw_value.replace("$", "")
    try:
        return Decimal(raw_value), is_percent, is_currency
    except InvalidOperation:
        return None


def _numbers_in(value: object) -> set[tuple[Decimal, bool, bool]]:
    return {
        signature
        for token in _NUMBER_PATTERN.findall(str(value))
        if (signature := _number_signature(token)) is not None
    }


def _allowed_numbers(item: EvidenceItem) -> set[tuple[Decimal, bool, bool]]:
    """Read numbers from the same canonical rendering used to create claims."""
    return _numbers_in(render_evidence_item(item))


def _base_reasons(
    explanation: GeneratedExplanation, evidence: ExplanationEvidence
) -> list[FaithfulnessReason]:
    reasons: list[FaithfulnessReason] = []
    if explanation.case_id != evidence.case_id:
        reasons.append(
            FaithfulnessReason(
                code=FaithfulnessReasonCode.CASE_ID_MISMATCH,
                message="Explanation case_id does not match evidence case_id.",
            )
        )
    trace = " ".join(claim.statement for claim in explanation.claims)
    if _normalise_text(explanation.text) != _normalise_text(trace):
        reasons.append(
            FaithfulnessReason(
                code=FaithfulnessReasonCode.TEXT_TRACE_MISMATCH,
                message="Explanation text does not match its rendered claim trace.",
            )
        )
    return reasons


def grade_prose_evidence(
    explanation: GeneratedExplanation, evidence: ExplanationEvidence
) -> FaithfulnessVerdict:
    """Check LLM prose against its closed claim/evidence trace without template matching.

    This complements, rather than replaces, ``grade_faithfulness``. The structural
    grader remains the deterministic regression gate; this grader validates typed LLM
    prose bindings, trace integrity, evidence coverage, and numeric provenance.
    """
    by_id = {item.evidence_id: item for item in evidence.items}
    reasons = _base_reasons(explanation, evidence)
    cited = tuple(claim.evidence_id for claim in explanation.claims)
    expected = tuple(item.evidence_id for item in evidence.items)
    if cited != expected:
        reasons.append(
            FaithfulnessReason(
                code=FaithfulnessReasonCode.CLAIM_COVERAGE_MISMATCH,
                message="Claim evidence IDs must exactly match supplied evidence order.",
            )
        )
    for index, claim in enumerate(explanation.claims):
        item = by_id.get(claim.evidence_id)
        if item is None:
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNKNOWN_EVIDENCE_ID,
                    claim_index=index,
                    evidence_id=claim.evidence_id,
                    message="Claim cites unknown evidence_id.",
                )
            )
            continue
        if claim.kind is not item.kind:
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.EVIDENCE_KIND_MISMATCH,
                    claim_index=index,
                    evidence_id=claim.evidence_id,
                    message="Claim kind does not match cited evidence kind.",
                )
            )
        if _normalise_text(claim.subject) != _normalise_text(item.label):
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNSUPPORTED_SUBJECT,
                    claim_index=index,
                    evidence_id=claim.evidence_id,
                    message="Claim subject is not the cited evidence label.",
                )
            )
        allowed_numbers = _allowed_numbers(item)
        for literal in _numbers_in(claim.statement):
            if literal not in allowed_numbers:
                reasons.append(
                    FaithfulnessReason(
                        code=FaithfulnessReasonCode.UNSUPPORTED_NUMBER,
                        claim_index=index,
                        evidence_id=claim.evidence_id,
                        message="Claim contains a numeric literal absent from cited evidence.",
                    )
                )
    return FaithfulnessVerdict(faithful=not reasons, reasons=tuple(reasons))


def grade_faithfulness(
    explanation: GeneratedExplanation, evidence: ExplanationEvidence
) -> FaithfulnessVerdict:
    """Apply the strict deterministic regression gate for canonical claim rendering."""
    by_id = {item.evidence_id: item for item in evidence.items}
    reasons = _base_reasons(explanation, evidence)
    cited = tuple(claim.evidence_id for claim in explanation.claims)
    expected = tuple(item.evidence_id for item in evidence.items)
    if cited != expected:
        reasons.append(
            FaithfulnessReason(
                code=FaithfulnessReasonCode.CLAIM_COVERAGE_MISMATCH,
                message="Claim evidence IDs must exactly match supplied evidence IDs in order.",
            )
        )
    for index, claim in enumerate(explanation.claims):
        item = by_id.get(claim.evidence_id)
        if item is None:
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNKNOWN_EVIDENCE_ID,
                    claim_index=index,
                    evidence_id=claim.evidence_id,
                    message="Claim cites unknown evidence_id.",
                )
            )
            continue
        if claim.kind is not item.kind:
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.EVIDENCE_KIND_MISMATCH,
                    claim_index=index,
                    evidence_id=claim.evidence_id,
                    message="Claim kind does not match cited evidence kind.",
                )
            )
        if _normalise_text(claim.subject) != _normalise_text(item.label):
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNSUPPORTED_SUBJECT,
                    claim_index=index,
                    evidence_id=claim.evidence_id,
                    message="Claim subject is not the cited evidence label.",
                )
            )
        if _normalise_text(claim.statement) != _normalise_text(render_evidence_item(item)):
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNSUPPORTED_STATEMENT,
                    claim_index=index,
                    evidence_id=claim.evidence_id,
                    message="Claim statement does not match canonical evidence rendering.",
                )
            )
        for literal in _numbers_in(claim.statement):
            if literal not in _allowed_numbers(item):
                reasons.append(
                    FaithfulnessReason(
                        code=FaithfulnessReasonCode.UNSUPPORTED_NUMBER,
                        claim_index=index,
                        evidence_id=claim.evidence_id,
                        message="Claim contains a numeric literal absent from cited evidence.",
                    )
                )
    return FaithfulnessVerdict(faithful=not reasons, reasons=tuple(reasons))
