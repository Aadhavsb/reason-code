"""Deterministic policy for turning a normalized score into a demo decision."""

from reason_code.schemas import Application, Decision, DecisionResponse, DecisionThresholds
from reason_code.scoring import CalibratedScoreProvider

DEFAULT_THRESHOLDS = DecisionThresholds(approve_below=0.40, decline_above=0.60)
POLICY_VERSION = "demo-thresholds-v1"
DEMO_NOTICE = (
    "Decision-support demo only. This result is not lending advice or production underwriting."
)


class DecisionService:
    """Applies one explicit threshold policy to an injected score provider."""

    def __init__(
        self,
        scorer: CalibratedScoreProvider,
        thresholds: DecisionThresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self._scorer = scorer
        self._thresholds = thresholds

    def decide(self, application: Application) -> DecisionResponse:
        result = self._scorer.score(application)
        if result.risk_score < self._thresholds.approve_below:
            decision = Decision.APPROVE
        elif result.risk_score > self._thresholds.decline_above:
            decision = Decision.DECLINE
        else:
            decision = Decision.HUMAN_REVIEW

        return DecisionResponse(
            application_id=application.application_id,
            decision=decision,
            risk_score=result.risk_score,
            score_source=result.score_source,
            attributions=result.attributions,
            policy_version=POLICY_VERSION,
            demo_notice=DEMO_NOTICE,
        )
