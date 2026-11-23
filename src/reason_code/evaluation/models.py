"""Typed contracts for optional, non-authoritative explanation judging."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from reason_code.explanations import ExplanationEvidence, GeneratedExplanation


class JudgeRequest(BaseModel):
    """One explanation evaluation, bound to immutable evidence and rubric identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: Annotated[str, Field(min_length=1)]
    evidence_digest: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    rubric_version: Annotated[str, Field(min_length=1)]
    explanation_text: Annotated[str, Field(min_length=1)]


class JudgeResponse(BaseModel):
    """A non-authoritative explanation-evaluation response.

    ``accepted`` means a recorded result matched the exact request binding. It does not
    affect generation, scoring, or the deterministic decision policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: Annotated[str, Field(min_length=1)]
    evidence_digest: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    rubric_version: Annotated[str, Field(min_length=1)]
    accepted: bool
    faithful: bool
    findings: tuple[str, ...] = ()
    source: Literal["recorded", "manual", "disabled"]
    synthetic: bool
    non_authoritative: bool = True


class ExplanationGenerator(Protocol):
    """Manual-only generator that receives the closed explanation evidence set."""

    def generate(self, evidence: ExplanationEvidence) -> GeneratedExplanation:
        """Return a typed trace or reject malformed/unfaithful output."""


class ExplanationJudge(Protocol):
    """Narrow seam for optional explanation evaluation only."""

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate an explanation without participating in decisioning."""
