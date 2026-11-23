"""Evidence-bounded explanation generation and faithfulness grading."""

from .generator import generate_explanation, render_evidence_item
from .grader import grade_faithfulness, grade_prose_evidence
from .models import (
    EvidenceItem,
    EvidenceKind,
    ExplanationClaim,
    ExplanationEvidence,
    FaithfulnessReason,
    FaithfulnessReasonCode,
    FaithfulnessVerdict,
    GeneratedExplanation,
)

__all__ = [
    "EvidenceItem",
    "EvidenceKind",
    "ExplanationClaim",
    "ExplanationEvidence",
    "FaithfulnessReason",
    "FaithfulnessReasonCode",
    "FaithfulnessVerdict",
    "GeneratedExplanation",
    "generate_explanation",
    "grade_faithfulness",
    "grade_prose_evidence",
    "render_evidence_item",
]
