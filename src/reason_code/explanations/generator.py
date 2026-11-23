"""Deterministic rendering of explanations from a closed evidence set."""

from __future__ import annotations

from decimal import Decimal

from .models import EvidenceItem, ExplanationClaim, ExplanationEvidence, GeneratedExplanation


def _render_value(value: str | int | float) -> str:
    """Render numeric values without non-canonical trailing zeroes or exponent notation."""
    if isinstance(value, float):
        decimal = Decimal(str(value))
        return format(decimal.normalize(), "f")
    return str(value)


def render_evidence_item(item: EvidenceItem) -> str:
    """Render one item without introducing facts not present in that item."""
    if item.value is not None:
        return f"{item.label}: {_render_value(item.value)}."
    if item.detail is not None:
        return f"{item.label}: {item.detail}."
    return f"{item.label}."


def generate_explanation(evidence: ExplanationEvidence) -> GeneratedExplanation:
    """Return a stable, evidence-only explanation and complete claim trace."""
    claims = tuple(
        ExplanationClaim(
            evidence_id=item.evidence_id,
            kind=item.kind,
            subject=item.label,
            statement=render_evidence_item(item),
        )
        for item in evidence.items
    )
    return GeneratedExplanation(
        case_id=evidence.case_id,
        text=" ".join(claim.statement for claim in claims),
        claims=claims,
    )
