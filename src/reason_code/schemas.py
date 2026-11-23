"""Validated contracts for the decision-support demo.

These schemas describe a demo workflow only. They are not a production underwriting
contract and do not establish lending eligibility.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
