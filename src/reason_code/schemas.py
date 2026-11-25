"""Validated contracts for the decision-support demo.

These schemas describe a deterministic demo workflow only. They are not a production
underwriting contract, do not establish lending eligibility, and do not claim HMDA
or other regulatory compliance.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reason_code.explanations import (
    EvidenceKind,
    ExplanationEvidence,
    FaithfulnessVerdict,
    GeneratedExplanation,
)
from reason_code.reason_mapping import DemoMappingEvidenceEnvelope, MappingStatus

DEMO_SCORE_SOURCE = "demo_heuristic_not_calibrated"
DEMO_EXPLANATION_NOTICE = (
    "Deterministic demo explanation only. Demo reason codes are not HMDA reason codes "
    "and do not establish production or regulatory compliance."
)


class Application(BaseModel):
    """Minimal application-shaped input accepted by the demo."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    application_id: str = Field(min_length=1, max_length=128)
    annual_income: float = Field(gt=0, description="Applicant-reported annual income.")
    requested_amount: float = Field(gt=0, description="Requested amount for the demo.")
    debt_to_income_ratio: float = Field(ge=0, le=100)
    credit_score: int = Field(ge=300, le=850)


class AttributionDirection(StrEnum):
    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"


class Attribution(BaseModel):
    """A scorer-supported factor contribution used in a decision response."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    factor: str = Field(min_length=1, max_length=64)
    direction: AttributionDirection
    contribution: float = Field(ge=0, le=1)
    detail: str = Field(min_length=1, max_length=280)


class ScoreResult(BaseModel):
    """Normalized score contract consumed by the decision policy.

    A production scorer could populate this model from a separately calibrated
    model. The bundled scorer labels its output as a demo heuristic instead.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    risk_score: float = Field(ge=0, le=1)
    score_source: str = Field(min_length=1, max_length=128)
    attributions: list[Attribution] = Field(default_factory=list)


class Decision(StrEnum):
    APPROVE = "approve"
    DECLINE = "decline"
    HUMAN_REVIEW = "human_review"


class DecisionResponse(BaseModel):
    """Decision-core output; no reason-code mapping or explanation is implied."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    application_id: str
    decision: Decision
    risk_score: float = Field(ge=0, le=1)
    score_source: str
    attributions: list[Attribution]
    policy_version: str
    demo_notice: str

    @model_validator(mode="after")
    def declines_require_supporting_attributions(self) -> "DecisionResponse":
        has_adverse_evidence = any(
            attribution.direction is AttributionDirection.INCREASES_RISK
            and attribution.contribution > 0
            for attribution in self.attributions
        )
        if self.decision is Decision.DECLINE and not has_adverse_evidence:
            raise ValueError(
                "decline decisions require at least one adverse, non-zero attribution"
            )
        return self


class DemoReasonCode(BaseModel):
    """A deterministic demo mapping from one adverse attribution to a demo code.

    This is deliberately not a HMDA, adverse-action, or production reason-code
    contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    code: str = Field(pattern=r"^demo_[a-z0-9_]+$")
    factor: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1, max_length=128)


class DecisionExplanationResponse(BaseModel):
    """Fail-closed deterministic explanation output for the bundled demo scorer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: DecisionResponse
    reason_codes: tuple[DemoReasonCode, ...]
    evidence: ExplanationEvidence
    mapping_envelope: DemoMappingEvidenceEnvelope
    explanation: GeneratedExplanation
    faithfulness: FaithfulnessVerdict
    explanation_notice: Literal[
        "Deterministic demo explanation only. Demo reason codes are not HMDA reason codes "
        "and do not establish production or regulatory compliance."
    ] = DEMO_EXPLANATION_NOTICE

    @model_validator(mode="after")
    def explanation_must_be_bound_to_faithful_demo_evidence(self) -> "DecisionExplanationResponse":
        if self.decision.score_source != DEMO_SCORE_SOURCE:
            raise ValueError("explanations are available only for the bundled demo heuristic")
        if self.evidence.case_id != self.decision.application_id:
            raise ValueError("evidence case_id must match decision application_id")
        if self.mapping_envelope.case_id != self.decision.application_id:
            raise ValueError("mapping envelope case_id must match decision application_id")
        if self.explanation.case_id != self.decision.application_id:
            raise ValueError("explanation case_id must match decision application_id")
        if not self.faithfulness.faithful:
            raise ValueError("faithfulness must pass before returning an explanation")

        reason_codes_by_factor = {
            reason_code.factor: reason_code for reason_code in self.reason_codes
        }
        if len(reason_codes_by_factor) != len(self.reason_codes):
            raise ValueError("reason codes must not repeat an attribution factor")
        if len({reason_code.code for reason_code in self.reason_codes}) != len(self.reason_codes):
            raise ValueError("reason codes must not repeat a code")
        if self.decision.decision is Decision.DECLINE and not self.reason_codes:
            raise ValueError("decline explanations require at least one reason code")

        adverse_factors = {
            attribution.factor
            for attribution in self.decision.attributions
            if attribution.direction is AttributionDirection.INCREASES_RISK
            and attribution.contribution > 0
        }
        if not set(reason_codes_by_factor).issubset(adverse_factors):
            raise ValueError("each reason code must reference an adverse attribution")
        evidenced_codes = [
            item.value
            for item in self.evidence.items
            if item.kind is EvidenceKind.REASON_CODE and isinstance(item.value, str)
        ]
        expected_codes = [reason_code.code for reason_code in self.reason_codes]
        if sorted(evidenced_codes) != sorted(expected_codes):
            raise ValueError("every reason code must be evidenced exactly once")
        if self.mapping_envelope.explanation_evidence != self.evidence:
            raise ValueError("mapping envelope must bind the returned evidence")
        if any(
            mapping.status is not MappingStatus.MAPPED
            for mapping in self.mapping_envelope.mappings
        ):
            raise ValueError("returned mapping envelope must contain only mapped factors")
        mapping_codes = [mapping.demo_code for mapping in self.mapping_envelope.mappings]
        if sorted(mapping_codes) != sorted(expected_codes):
            raise ValueError("mapping envelope codes must match returned reason codes")
        return self


class DecisionThresholds(BaseModel):
    """Risk-score policy with an inclusive human-review confidence band."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    approve_below: float = Field(ge=0, le=1)
    decline_above: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def review_band_must_have_width(self) -> "DecisionThresholds":
        if self.approve_below >= self.decline_above:
            raise ValueError("approve_below must be less than decline_above")
        return self
