"""Pydantic domain models for evidence-bounded explanations."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceKind(StrEnum):
    """The only kinds of evidence an explanation may cite."""

    DECISION = "decision"
    FACTOR = "factor"
    REASON_CODE = "reason_code"
    METRIC = "metric"


class EvidenceItem(BaseModel):
    """A single auditable item made available to an explanation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    evidence_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    kind: EvidenceKind
    label: Annotated[str, Field(min_length=1)]
    value: str | int | float | None = None
    detail: str | None = None


class ExplanationEvidence(BaseModel):
    """The closed evidence set for one explanation."""

    model_config = ConfigDict(frozen=True)

    case_id: Annotated[str, Field(min_length=1)]
    items: Annotated[tuple[EvidenceItem, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> ExplanationEvidence:
        evidence_ids = [item.evidence_id for item in self.items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique within an evidence set")
        return self


class ExplanationClaim(BaseModel):
    """A claim and the evidence item it asserts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    evidence_id: str
    kind: EvidenceKind
    subject: Annotated[str, Field(min_length=1)]
    statement: Annotated[str, Field(min_length=1)]


class GeneratedExplanation(BaseModel):
    """A rendered explanation plus its machine-readable claim trace."""

    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1)]
    claims: Annotated[tuple[ExplanationClaim, ...], Field(min_length=1)]


class FaithfulnessReasonCode(StrEnum):
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    EVIDENCE_KIND_MISMATCH = "evidence_kind_mismatch"
    UNSUPPORTED_SUBJECT = "unsupported_subject"
    UNSUPPORTED_NUMBER = "unsupported_number"


class FaithfulnessReason(BaseModel):
    """A single deterministic explanation-faithfulness finding."""

    model_config = ConfigDict(frozen=True)

    code: FaithfulnessReasonCode
    claim_index: int
    evidence_id: str
    message: str


class FaithfulnessVerdict(BaseModel):
    """Machine-readable result of grading one explanation against its evidence."""

    model_config = ConfigDict(frozen=True)

    faithful: bool
    reasons: tuple[FaithfulnessReason, ...]
