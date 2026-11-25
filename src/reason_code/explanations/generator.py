"""Deterministic rendering of explanations from a closed evidence set."""

from __future__ import annotations

from decimal import Decimal

from .models import EvidenceItem, ExplanationClaim, ExplanationEvidence, GeneratedExplanation


def render_value(value: str | int | float) -> str:
    """Render a value with the canonical numeric spelling used by all graders.

    Finite floats are converted through ``str`` before Decimal construction so their
    human-facing spelling does not expose binary floating-point artifacts. Decimal's
    fixed-point rendering also makes scientific notation stable (for example,
    ``1e-07`` becomes ``0.0000001``).
    """
    if isinstance(value, float):
        decimal = Decimal(str(value))
        return format(decimal.normalize(), "f")
    return str(value)


def render_evidence_item(item: EvidenceItem) -> str:
    """Render one item without introducing facts not present in that item."""
    if item.value is not None:
        return f"{item.label}: {render_value(item.value)}."
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
