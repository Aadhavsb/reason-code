from fastapi.testclient import TestClient

from reason_code.api import app, get_decision_service
from reason_code.decision import DecisionService
from reason_code.schemas import ScoreResult


class InjectedScorer:
    def score(self, application):
        return ScoreResult(risk_score=0.61, score_source="injected_test_scorer")


def valid_payload() -> dict[str, object]:
    return {
        "application_id": "application-123",
        "annual_income": 100_000,
        "requested_amount": 20_000,
        "debt_to_income_ratio": 25,
        "credit_score": 720,
    }


def test_post_decisions_uses_injected_service() -> None:
    app.dependency_overrides[get_decision_service] = lambda: DecisionService(InjectedScorer())
    try:
        response = TestClient(app).post("/decisions", json=valid_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["decision"] == "decline"
    assert response.json()["score_source"] == "injected_test_scorer"


def test_post_decisions_rejects_invalid_application() -> None:
    payload = valid_payload()
    payload["credit_score"] = 900
    payload["unexpected"] = "not accepted"

    response = TestClient(app).post("/decisions", json=payload)

    assert response.status_code == 422
