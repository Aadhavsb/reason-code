"""Evidence-bounded explanation generation and faithfulness grading."""

from .generator import generate_explanation
from .grader import grade_faithfulness
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
]
