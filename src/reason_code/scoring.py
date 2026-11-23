"""Injectable score providers for the decision-support demo."""

from typing import Protocol

from reason_code.schemas import Application, Attribution, AttributionDirection, ScoreResult


class CalibratedScoreProvider(Protocol):
    """Provides a normalized risk score and its supporting factor attributions.

    Implementations backed by a calibrated model should identify that model in
    ``score_source``. The bundled implementation deliberately does not claim
    calibration or HMDA-model performance.
    """

    def score(self, application: Application) -> ScoreResult:
        """Return a normalized score in the validated ScoreResult contract."""


class DemoHeuristicScorer:
    """Small deterministic scorer for wiring and testable demo behavior only."""

    score_source = "demo_heuristic_not_calibrated"

    def score(self, application: Application) -> ScoreResult:
        income_ratio = min(application.requested_amount / application.annual_income, 2.0) / 2.0
        dti_component = application.debt_to_income_ratio / 100 * 0.50
        amount_component = income_ratio * 0.30
        credit_component = (850 - application.credit_score) / 550 * 0.20
        risk_score = dti_component + amount_component + credit_component

        return ScoreResult(
            risk_score=risk_score,
            score_source=self.score_source,
            attributions=[
                Attribution(
                    factor="debt_to_income_ratio",
                    direction=AttributionDirection.INCREASES_RISK,
                    contribution=dti_component,
                    detail="Demo heuristic contribution from debt-to-income ratio.",
                ),
                Attribution(
                    factor="requested_amount_to_income",
                    direction=AttributionDirection.INCREASES_RISK,
                    contribution=amount_component,
                    detail="Demo heuristic contribution from requested amount relative to income.",
                ),
                Attribution(
                    factor="credit_score",
                    direction=AttributionDirection.INCREASES_RISK,
                    contribution=credit_component,
                    detail=(
                        "Demo heuristic contribution from credit score distance to the upper bound."
                    ),
                ),
            ],
        )
