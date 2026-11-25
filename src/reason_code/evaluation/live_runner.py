"""Narrow, manually invoked one-request experiment over one verified synthetic fixture."""

from __future__ import annotations

import hashlib
from typing import Any

from reason_code.explanations import ExplanationEvidence, generate_explanation

from .config import JudgeMode, JudgeSettings
from .manual import (
    LlmMalformedResponseError,
    LlmProviderRequestError,
    LlmSchemaRejectedError,
    LlmUnfaithfulOutputError,
    ManualExplanationGenerator,
    ManualLlmClient,
    Transport,
)
from .runner import EvaluationError, load_verified_fixture

LIVE_FIXTURE_CASE_ID = "synthetic-faithful-001"
LIVE_EXPERIMENT_VERSION = "manual-live-experiment-v1"
LIVE_REQUEST_CAP = 1


class LiveFixtureUnavailableError(ValueError):
    """The named verified synthetic fixture is absent after checksum validation."""


def _fixture_evidence() -> tuple[ExplanationEvidence, str]:
    """Select one case only after the canonical deterministic validation/checksum pass."""
    cases, fixture_hash = load_verified_fixture()
    matches = [
        case["evidence"] for case in cases if case["evidence"].case_id == LIVE_FIXTURE_CASE_ID
    ]
    if len(matches) != 1:
        raise LiveFixtureUnavailableError("named live synthetic fixture is unavailable")
    return matches[0], fixture_hash


def minimum_output_token_cap(evidence: ExplanationEvidence) -> int:
    """Return a conservative byte-token floor for the deterministic valid output shape.

    JSON output can tokenize as finely as one token per UTF-8 byte. Using the canonical
    deterministic explanation's byte length makes a cap below this floor impossible by
    construction, while the settings ceiling still bounds local exposure.
    """
    return len(generate_explanation(evidence).model_dump_json().encode("utf-8"))


def _metadata(settings: JudgeSettings, fixture_hash: str, minimum_cap: int) -> dict[str, object]:
    return {
        "endpoint": settings.resolved_endpoint,
        "experiment_version": LIVE_EXPERIMENT_VERSION,
        "fixture_case_id": LIVE_FIXTURE_CASE_ID,
        "fixture_sha256": fixture_hash,
        "model": settings.model,
        "minimum_output_token_cap": minimum_cap,
        "n": 1,
        "non_authoritative": True,
        "output_token_cap": settings.output_token_cap,
        "provider": settings.provider.value,
        "request_cap": LIVE_REQUEST_CAP,
        "timeout_seconds": settings.timeout_seconds,
    }


def _failed_report(metadata: dict[str, object], cause: str, finding: str) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "rejection_cause": cause,
        "status": "fail",
        "verdict": {"accepted": False, "faithful": False, "findings": [finding]},
    }


def run_live_experiment(
    settings: JudgeSettings,
    transport: Transport | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make at most one request and return an explicitly non-authoritative n=1 report."""
    if settings.mode is not JudgeMode.MANUAL:
        raise ValueError("live experiments require manual judge mode")
    try:
        evidence, fixture_hash = _fixture_evidence()
    except EvaluationError:
        metadata = _metadata(settings, "unavailable", 0)
        return _failed_report(
            metadata,
            "fixture_integrity",
            "Verified synthetic fixture integrity check failed; no request was made.",
        )
    except LiveFixtureUnavailableError:
        metadata = _metadata(settings, "unavailable", 0)
        return _failed_report(
            metadata,
            "fixture_integrity",
            "Named verified synthetic fixture is unavailable; no request was made.",
        )
    minimum_cap = minimum_output_token_cap(evidence)
    metadata = _metadata(settings, fixture_hash, minimum_cap)
    if settings.output_token_cap < minimum_cap:
        return _failed_report(
            metadata,
            "configuration",
            "Output token cap is below the evidence-dependent minimum; no request was made.",
        )
    client = ManualLlmClient(settings, transport=transport, environ=environ)
    try:
        explanation = ManualExplanationGenerator(client).generate(evidence)
    except LlmMalformedResponseError:
        return _failed_report(metadata, "malformed_output", "Live output was malformed.")
    except LlmSchemaRejectedError:
        return _failed_report(
            metadata, "schema_rejection", "Live output failed typed schema validation."
        )
    except LlmUnfaithfulOutputError:
        return _failed_report(
            metadata, "unfaithfulness", "Live output failed deterministic evidence validation."
        )
    except LlmProviderRequestError:
        return _failed_report(metadata, "provider_failure", "Live provider request failed.")

    explanation_hash = hashlib.sha256(explanation.model_dump_json().encode("utf-8")).hexdigest()
    return {
        "metadata": metadata,
        "output_sha256": explanation_hash,
        "status": "pass",
        "verdict": {"accepted": True, "faithful": True, "findings": []},
    }
