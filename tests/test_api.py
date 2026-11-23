import asyncio
import json
import math

import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from reason_code import api as api_module
from reason_code.api import app, get_decision_service, request_validation_exception_handler
from reason_code.decision import DecisionService
from reason_code.schemas import Attribution, AttributionDirection, ScoreResult


class InjectedScorer:
    def score(self, application):
        return ScoreResult(
            risk_score=0.61,
            score_source="injected_test_scorer",
            attributions=[
                Attribution(
                    factor="test_factor",
                    direction=AttributionDirection.INCREASES_RISK,
                    contribution=0.61,
                    detail="Test-only supporting factor.",
                )
            ],
        )


def valid_payload() -> dict[str, object]:
    return {
        "application_id": "application-123",
        "annual_income": 100_000,
        "requested_amount": 20_000,
        "debt_to_income_ratio": 25,
        "credit_score": 720,
    }


def test_local_demo_page_is_self_contained_and_labels_its_limits() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Decision-support demo only." in response.text
    assert "No HMDA dataset is loaded or represented here." in response.text
    assert "Not computed by <code>/decisions</code>" in response.text
    assert '<script src=' not in response.text
    assert '<link href=' not in response.text
    assert "@import" not in response.text
    assert "fetch('/decisions'" in response.text


def test_local_demo_returns_json_error_when_ui_asset_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "_UI_PATH", api_module._UI_PATH.with_name("missing.html"))

    response = TestClient(app, raise_server_exceptions=False).get("/")

    assert response.status_code == 503
    assert response.json() == {"detail": "Local demo UI asset is unavailable."}


def test_post_decisions_uses_injected_service() -> None:
    app.dependency_overrides[get_decision_service] = lambda: DecisionService(InjectedScorer())
    try:
        response = TestClient(app).post("/decisions", json=valid_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["decision"] == "decline"
    assert response.json()["score_source"] == "injected_test_scorer"


def test_validation_handler_preserves_framework_encoding_for_validator_errors() -> None:
    request = Request({"type": "http", "method": "POST", "path": "/decisions", "headers": []})
    error = RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body", "application"),
                "msg": "Value error, invalid application",
                "input": {},
                "ctx": {"error": ValueError("invalid application")},
            }
        ]
    )

    response = asyncio.run(request_validation_exception_handler(request, error))

    assert response.status_code == 422
    assert b'"error":{}' in response.body


def test_post_decisions_rejects_invalid_application() -> None:
    payload = valid_payload()
    payload["credit_score"] = 900
    payload["unexpected"] = "not accepted"

    response = TestClient(app).post("/decisions", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_post_decisions_rejects_non_finite_numbers_as_validation_errors(non_finite: float) -> None:
    payload = valid_payload()
    payload["annual_income"] = non_finite

    response = TestClient(app).post(
        "/decisions", content=json.dumps(payload), headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "finite_number"
