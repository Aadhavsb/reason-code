import math

from reason_code.schemas import Application, AttributionDirection
from reason_code.scoring import DemoHeuristicScorer


def application(**overrides: float | int | str) -> Application:
    values: dict[str, float | int | str] = {
        "application_id": "application-123",
        "annual_income": 100_000,
        "requested_amount": 20_000,
        "debt_to_income_ratio": 25,
        "credit_score": 720,
    }
    values.update(overrides)
    return Application(**values)


def test_demo_heuristic_outputs_are_finite_and_normalized_at_input_bounds() -> None:
    scorer = DemoHeuristicScorer()
    applications = (
        application(),
        application(
            annual_income=0.01,
            requested_amount=1_000_000,
            debt_to_income_ratio=100,
            credit_score=300,
        ),
        application(
            annual_income=1_000_000, requested_amount=0.01, debt_to_income_ratio=0, credit_score=850
        ),
    )

    for candidate in applications:
        result = scorer.score(candidate)

        assert math.isfinite(result.risk_score)
        assert 0 <= result.risk_score <= 1
        assert all(math.isfinite(attribution.contribution) for attribution in result.attributions)
        assert all(0 <= attribution.contribution <= 1 for attribution in result.attributions)


def test_demo_heuristic_factor_directions_match_risk_semantics() -> None:
    scorer = DemoHeuristicScorer()
    lower_credit = scorer.score(application(credit_score=600))
    higher_credit = scorer.score(application(credit_score=800))
    directions = {
        attribution.factor: attribution.direction for attribution in higher_credit.attributions
    }

    assert lower_credit.risk_score > higher_credit.risk_score
    assert directions == {
        "debt_to_income_ratio": AttributionDirection.INCREASES_RISK,
        "requested_amount_to_income": AttributionDirection.INCREASES_RISK,
        "credit_score": AttributionDirection.DECREASES_RISK,
    }
