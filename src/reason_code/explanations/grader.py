"""Programmatic, deterministic faithfulness checks for explanation claim traces."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .generator import _render_item
from .models import (
    EvidenceItem,
    ExplanationEvidence,
    FaithfulnessReason,
    FaithfulnessReasonCode,
    FaithfulnessVerdict,
    GeneratedExplanation,
)

_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\$?\d[\d,]*(?:\.\d+)?%?")


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _number_signature(token: str) -> tuple[Decimal, bool] | None:
    """Normalise a numeric literal while retaining whether it is a percentage."""
    is_percent = token.endswith("%")
    raw_value = token.removesuffix("%").removeprefix("$").replace(",", "")
    try:
        return Decimal(raw_value), is_percent
    except InvalidOperation:
        return None


def _numbers_in(value: object) -> set[tuple[Decimal, bool]]:
    return {
        signature
        for token in _NUMBER_PATTERN.findall(str(value))
        if (signature := _number_signature(token)) is not None
    }


def _allowed_numbers(item: EvidenceItem) -> set[tuple[Decimal, bool]]:
    """Collect literals explicitly present in the cited evidence item only."""
    values: list[object] = [item.label]
    if item.value is not None:
        values.append(item.value)
    if item.detail is not None:
        values.append(item.detail)
    return set().union(*(_numbers_in(value) for value in values))


def grade_faithfulness(
    explanation: GeneratedExplanation, evidence: ExplanationEvidence
) -> FaithfulnessVerdict:
    """Grade each claim only against the item it cites.

    This checker intentionally evaluates structured claims rather than attempting to
    infer provenance from free prose. A claim must cite an evidence ID, use its exact
    kind and label, and use only numeric literals present in that same evidence item.
    """
    by_id = {item.evidence_id: item for item in evidence.items}
    reasons: list[FaithfulnessReason] = []

    for claim_index, claim in enumerate(explanation.claims):
        item = by_id.get(claim.evidence_id)
        if item is None:
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNKNOWN_EVIDENCE_ID,
                    claim_index=claim_index,
                    evidence_id=claim.evidence_id,
                    message=f"Claim cites unknown evidence_id '{claim.evidence_id}'.",
                )
            )
            continue

        if claim.kind is not item.kind:
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.EVIDENCE_KIND_MISMATCH,
                    claim_index=claim_index,
                    evidence_id=claim.evidence_id,
                    message=(
                        f"Claim kind '{claim.kind.value}' does not match cited "
                        f"evidence kind '{item.kind.value}'."
                    ),
                )
            )

        if _normalise_text(claim.subject) != _normalise_text(item.label):
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNSUPPORTED_SUBJECT,
                    claim_index=claim_index,
                    evidence_id=claim.evidence_id,
                    message=(
                        f"Claim subject '{claim.subject}' is not the cited evidence "
                        f"label '{item.label}'."
                    ),
                )
            )

        expected_statement = _render_item(item)
        if _normalise_text(claim.statement) != _normalise_text(expected_statement):
            reasons.append(
                FaithfulnessReason(
                    code=FaithfulnessReasonCode.UNSUPPORTED_STATEMENT,
                    claim_index=claim_index,
                    evidence_id=claim.evidence_id,
                    message=(
                        "Claim statement does not match the deterministic rendering of "
                        f"evidence_id '{claim.evidence_id}'."
                    ),
                )
            )

        allowed_numbers = _allowed_numbers(item)
        for numeric_literal in _numbers_in(claim.statement):
            if numeric_literal not in allowed_numbers:
                reasons.append(
                    FaithfulnessReason(
                        code=FaithfulnessReasonCode.UNSUPPORTED_NUMBER,
                        claim_index=claim_index,
                        evidence_id=claim.evidence_id,
                        message=(
                            f"Claim contains unsupported numeric literal "
                            f"'{numeric_literal[0]}' for evidence_id '{claim.evidence_id}'."
                        ),
                    )
                )

    return FaithfulnessVerdict(faithful=not reasons, reasons=tuple(reasons))
