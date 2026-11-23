"""Deterministic rendering of explanations from a closed evidence set."""

from __future__ import annotations

from .models import EvidenceItem, ExplanationClaim, ExplanationEvidence, GeneratedExplanation


def _render_item(item: EvidenceItem) -> str:
    """Render one item without introducing facts not present in that item."""
    if item.value is not None:
        return f"{item.label}: {item.value}."
    if item.detail:
        return f"{item.label}: {item.detail}."
    return f"{item.label}."


def generate_explanation(evidence: ExplanationEvidence) -> GeneratedExplanation:
    """Return a stable, evidence-only explanation and complete claim trace.

    Input order is retained deliberately: upstream decisioning controls the presentation
    order, and the same evidence produces byte-for-byte identical output.
    """
    claims = tuple(
        ExplanationClaim(
            evidence_id=item.evidence_id,
            kind=item.kind,
            subject=item.label,
            statement=_render_item(item),
        )
        for item in evidence.items
    )
    return GeneratedExplanation(
        text=" ".join(claim.statement for claim in claims),
        claims=claims,
    )
