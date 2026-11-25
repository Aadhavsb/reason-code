import asyncio
import json
import math

import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from reason_code import api as api_module
from reason_code.api import app, get_decision_service, request_validation_exception_handler
from reason_code.decision import DecisionService
from reason_code.schemas import (
    Attribution,
    AttributionDirection,
    DecisionExplanationResponse,
    ScoreResult,
)


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


class UnmappedDemoScorer:
    def score(self, application):
        return ScoreResult(
            risk_score=0.61,
            score_source="demo_heuristic_not_calibrated",
            attributions=[
                Attribution(
                    factor="unmapped_factor",
                    direction=AttributionDirection.INCREASES_RISK,
                    contribution=0.61,
                    detail="Test-only unmapped factor.",
                )
            ],
        )


class DuplicateAdverseFactorScorer:
    def score(self, application):
        attribution = Attribution(
            factor="debt_to_income_ratio",
            direction=AttributionDirection.INCREASES_RISK,
            contribution=0.31,
            detail="Test-only duplicate adverse attribution.",
        )
        return ScoreResult(
            risk_score=0.62,
            score_source="demo_heuristic_not_calibrated",
            attributions=[attribution, attribution],
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
    assert "<code>/decisions/explanation</code> endpoint" in response.text
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


def test_explanation_refusal_is_documented_as_unprocessable_content() -> None:
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == 200
    refusal = response.json()["paths"]["/decisions/explanation"]["post"]["responses"]["422"]
    assert "deterministic explanation path refuses" in refusal["description"]


def test_post_decision_explanation_returns_verified_demo_evidence() -> None:
    response = TestClient(app).post("/decisions/explanation", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["score_source"] == "demo_heuristic_not_calibrated"
    assert body["faithfulness"] == {"faithful": True, "reasons": []}
    assert "prose_faithfulness" not in body
    assert body["evidence"]["case_id"] == valid_payload()["application_id"]
    assert body["mapping_envelope"]["case_id"] == valid_payload()["application_id"]
    assert body["mapping_envelope"]["contract_version"] == "demo-evidence-reason-mapping-v1"
    assert body["mapping_envelope"]["catalog_version"] == "demo-factor-catalog-v1"
    assert body["mapping_envelope"]["explanation_evidence"] == body["evidence"]
    assert {mapping["status"] for mapping in body["mapping_envelope"]["mappings"]} == {"mapped"}
    assert body["explanation"]["case_id"] == valid_payload()["application_id"]
    assert [item["evidence_id"] for item in body["evidence"]["items"]] == [
        "decision",
        "factor_debt_to_income_ratio",
        "reason_debt_to_income_ratio",
        "factor_requested_amount_to_income",
        "reason_requested_amount_to_income",
        "factor_credit_score_deficiency",
        "reason_credit_score_deficiency",
    ]
    assert [reason["code"] for reason in body["reason_codes"]] == [
        "demo_debt_to_income_risk_contribution",
        "demo_requested_amount_to_income_risk_contribution",
        "demo_credit_score_deficiency_risk_contribution",
    ]
    assert "not HMDA reason codes" in body["explanation_notice"]
    assert "production or regulatory compliance" in body["explanation_notice"]


def test_post_decision_explanation_decline_has_evidenced_reason_codes() -> None:
    payload = {
        "application_id": "application-decline-123",
        "annual_income": 50_000,
        "requested_amount": 90_000,
        "debt_to_income_ratio": 85,
        "credit_score": 500,
    }

    response = TestClient(app).post("/decisions/explanation", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["decision"] == "decline"
    assert body["reason_codes"]
    evidenced_codes = {
        item["value"] for item in body["evidence"]["items"] if item["kind"] == "reason_code"
    }
    assert evidenced_codes == {reason_code["code"] for reason_code in body["reason_codes"]}


def test_explanation_response_rejects_unbacked_reason_codes_and_empty_declines() -> None:
    payload = {
        "application_id": "application-decline-456",
        "annual_income": 50_000,
        "requested_amount": 90_000,
        "debt_to_income_ratio": 85,
        "credit_score": 500,
    }
    response = TestClient(app).post("/decisions/explanation", json=payload)
    assert response.status_code == 200

    missing_reason_codes = response.json()
    missing_reason_codes["reason_codes"] = []
    with pytest.raises(
        ValidationError, match="decline explanations require at least one reason code"
    ):
        DecisionExplanationResponse.model_validate(missing_reason_codes)

    missing_reason_evidence = response.json()
    missing_reason_evidence["evidence"]["items"] = [
        item
        for item in missing_reason_evidence["evidence"]["items"]
        if item["kind"] != "reason_code"
    ]
    with pytest.raises(ValidationError, match="every reason code must be evidenced exactly once"):
        DecisionExplanationResponse.model_validate(missing_reason_evidence)


def test_post_decision_explanation_fails_closed_for_non_demo_scorer() -> None:
    app.dependency_overrides[get_decision_service] = lambda: DecisionService(InjectedScorer())
    try:
        response = TestClient(app).post("/decisions/explanation", json=valid_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Deterministic explanations are available only for the bundled demo heuristic."
    }


def test_post_decision_explanation_fails_closed_for_unmapped_adverse_factor() -> None:
    app.dependency_overrides[get_decision_service] = lambda: DecisionService(UnmappedDemoScorer())
    try:
        response = TestClient(app).post("/decisions/explanation", json=valid_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {
        "detail": "No deterministic demo reason-code mapping exists for an adverse attribution."
    }


def test_post_decision_explanation_refuses_duplicate_adverse_factor() -> None:
    app.dependency_overrides[get_decision_service] = lambda: DecisionService(
        DuplicateAdverseFactorScorer()
    )
    try:
        response = TestClient(app).post("/decisions/explanation", json=valid_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "Duplicate adverse attribution factors cannot be represented in a "
            "deterministic explanation."
        )
    }


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
