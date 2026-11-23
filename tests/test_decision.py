import pytest
from pydantic import ValidationError

from reason_code.decision import DecisionService
from reason_code.schemas import (
    Application,
    Attribution,
    AttributionDirection,
    Decision,
    DecisionThresholds,
    ScoreResult,
)


class FixedScorer:
    def __init__(self, risk_score: float, attributions: list[Attribution] | None = None) -> None:
        self.risk_score = risk_score
        self.attributions = attributions or []

    def score(self, application: Application) -> ScoreResult:
        return ScoreResult(
            risk_score=self.risk_score,
            score_source="test_calibrated_score",
            attributions=self.attributions,
        )


def application() -> Application:
    return Application(
        application_id="application-123",
        annual_income=100_000,
        requested_amount=20_000,
        debt_to_income_ratio=25,
        credit_score=720,
    )


def supporting_attribution() -> Attribution:
    return Attribution(
        factor="test_factor",
        direction=AttributionDirection.INCREASES_RISK,
        contribution=0.7,
        detail="Test-only supporting factor.",
    )


def test_threshold_policy_routes_low_middle_and_high_scores() -> None:
    decisions = {
        score: DecisionService(
            FixedScorer(score, [supporting_attribution()] if score > 0.60 else [])
        ).decide(application()).decision
        for score in (0.39, 0.40, 0.60, 0.61)
    }

    assert decisions == {
        0.39: Decision.APPROVE,
        0.40: Decision.HUMAN_REVIEW,
        0.60: Decision.HUMAN_REVIEW,
        0.61: Decision.DECLINE,
    }


def test_service_preserves_scorer_attributions() -> None:
    response = DecisionService(FixedScorer(0.39)).decide(application())

    assert response.application_id == "application-123"
    assert response.risk_score == 0.39
    assert response.score_source == "test_calibrated_score"
    assert response.attributions == []
    assert "not lending advice" in response.demo_notice


def test_decline_requires_adverse_nonzero_supporting_attribution() -> None:
    with pytest.raises(ValidationError, match="adverse, non-zero"):
        DecisionService(FixedScorer(0.61)).decide(application())

    non_supporting = Attribution(
        factor="risk_reducer",
        direction=AttributionDirection.DECREASES_RISK,
        contribution=0,
        detail="A zero-magnitude favorable factor is not adverse decline evidence.",
    )
    with pytest.raises(ValidationError, match="adverse, non-zero"):
        DecisionService(FixedScorer(0.61, [non_supporting])).decide(application())


def test_thresholds_are_immutable_and_not_shared_between_default_services() -> None:
    first = DecisionService(FixedScorer(0.39))
    second = DecisionService(FixedScorer(0.39))

    assert first._thresholds is not second._thresholds
    with pytest.raises(ValidationError, match="frozen"):
        first._thresholds.approve_below = 0.20


def test_policy_version_is_bound_to_applied_threshold_values() -> None:
    response = DecisionService(
        FixedScorer(0.50),
        thresholds=DecisionThresholds(approve_below=0.25, decline_above=0.75),
    ).decide(application())

    assert response.policy_version == "demo-thresholds-v1:approve_below=0.25;decline_above=0.75"
