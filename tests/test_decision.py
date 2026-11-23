from reason_code.decision import DecisionService
from reason_code.schemas import Application, Decision, ScoreResult


class FixedScorer:
    def __init__(self, risk_score: float) -> None:
        self.risk_score = risk_score

    def score(self, application: Application) -> ScoreResult:
        return ScoreResult(risk_score=self.risk_score, score_source="test_calibrated_score")


def application() -> Application:
    return Application(
        application_id="application-123",
        annual_income=100_000,
        requested_amount=20_000,
        debt_to_income_ratio=25,
        credit_score=720,
    )


def test_threshold_policy_routes_low_middle_and_high_scores() -> None:
    decisions = {
        score: DecisionService(FixedScorer(score)).decide(application()).decision
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
