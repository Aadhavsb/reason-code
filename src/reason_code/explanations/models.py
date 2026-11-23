"""Pydantic domain models for evidence-bounded explanations."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceKind(StrEnum):
    """The only kinds of evidence an explanation may cite."""

    DECISION = "decision"
    FACTOR = "factor"
    REASON_CODE = "reason_code"
    METRIC = "metric"


class EvidenceItem(BaseModel):
    """A single auditable item made available to an explanation."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, allow_inf_nan=False
    )

    evidence_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    kind: EvidenceKind
    label: Annotated[str, Field(min_length=1)]
    value: str | int | float | None = None
    detail: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def value_must_be_renderable(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("evidence value must not be a boolean")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("evidence value must be finite")
        if isinstance(value, str) and not value.strip():
            raise ValueError("evidence value must not be empty")
        return value

    @field_validator("detail", mode="before")
    @classmethod
    def detail_must_not_be_empty(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("evidence detail must not be empty")
        return value


class ExplanationEvidence(BaseModel):
    """The closed evidence set for one explanation."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

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

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    evidence_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    kind: EvidenceKind
    subject: Annotated[str, Field(min_length=1)]
    statement: Annotated[str, Field(min_length=1)]


class GeneratedExplanation(BaseModel):
    """A rendered explanation plus its machine-readable claim trace."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    case_id: Annotated[str, Field(min_length=1)]
    text: Annotated[str, Field(min_length=1)]
    claims: Annotated[tuple[ExplanationClaim, ...], Field(min_length=1)]


class FaithfulnessReasonCode(StrEnum):
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    EVIDENCE_KIND_MISMATCH = "evidence_kind_mismatch"
    UNSUPPORTED_SUBJECT = "unsupported_subject"
    UNSUPPORTED_STATEMENT = "unsupported_statement"
    UNSUPPORTED_NUMBER = "unsupported_number"
    TEXT_TRACE_MISMATCH = "text_trace_mismatch"
    CASE_ID_MISMATCH = "case_id_mismatch"
    CLAIM_COVERAGE_MISMATCH = "claim_coverage_mismatch"


class FaithfulnessReason(BaseModel):
    """A single deterministic explanation-faithfulness finding."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    code: FaithfulnessReasonCode
    claim_index: int | None = None
    evidence_id: str | None = None
    message: Annotated[str, Field(min_length=1)]


class FaithfulnessVerdict(BaseModel):
    """Machine-readable result of grading one explanation against its evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    faithful: bool
    reasons: tuple[FaithfulnessReason, ...]

    @model_validator(mode="after")
    def faithful_must_match_reasons(self) -> FaithfulnessVerdict:
        if self.faithful != (not self.reasons):
            raise ValueError("faithful must be true exactly when reasons is empty")
        return self
