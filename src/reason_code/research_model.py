"""JSON-only artifact support for the offline HMDA research model.

This module is intentionally separate from the demo decision service.  It scores only
three numeric research features and returns a denial-association probability; it never
returns lending decisions, reason codes, or explanations.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from reason_code.provenance import parse_json_object

ARTIFACT_SCHEMA_VERSION = "1.0"
MODEL_VERSION = "hmda-logistic-platt-v1"
FEATURES = ("annual_income", "requested_amount", "debt_to_income_ratio")


class ModelArtifactValidationError(ValueError):
    """Raised when a local research-model artifact is malformed or altered."""


@dataclass(frozen=True)
class LocalChronologicalModel:
    """Validated, inference-only representation of the research logistic+Platt model."""

    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    logistic_coefficients: tuple[float, ...]
    logistic_intercept: float
    platt_coefficient: float
    platt_intercept: float

    def score_features(self, values: Mapping[str, float | int]) -> float:
        """Return calibrated denial-association probability for exactly the artifact features."""
        if set(values) != set(FEATURES):
            raise ValueError("values must contain exactly the artifact feature names")
        numeric_values = tuple(_finite_number(values[name], f"values.{name}") for name in FEATURES)
        linear = self.logistic_intercept + sum(
            coefficient * ((value - mean) / scale)
            for value, mean, scale, coefficient in zip(
                numeric_values,
                self.feature_means,
                self.feature_scales,
                self.logistic_coefficients,
                strict=True,
            )
        )
        raw_probability = _sigmoid(linear)
        return _sigmoid(self.platt_intercept + self.platt_coefficient * raw_probability)


def _canonical_sha256(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ModelArtifactValidationError(f"{field} must be a finite JSON number")
    return float(value)


def _finite_vector(value: object, field: str, *, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(FEATURES):
        raise ModelArtifactValidationError(f"{field} must be a {len(FEATURES)}-item JSON array")
    result = tuple(_finite_number(item, f"{field}[{index}]") for index, item in enumerate(value))
    if positive and any(item <= 0 for item in result):
        raise ModelArtifactValidationError(f"{field} must contain only positive values")
    return result


def _object(value: object, field: str, expected_fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelArtifactValidationError(f"{field} must be a JSON object")
    missing = expected_fields.difference(value)
    unknown = set(value).difference(expected_fields)
    if missing or unknown:
        raise ModelArtifactValidationError(
            f"{field} must contain exactly: {', '.join(sorted(expected_fields))}"
        )
    return value


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def artifact_document(
    *,
    feature_means: Sequence[float],
    feature_scales: Sequence[float],
    logistic_coefficients: Sequence[float],
    logistic_intercept: float,
    platt_coefficient: float,
    platt_intercept: float,
) -> dict[str, object]:
    """Create a checksummed JSON-only document from fitted sklearn parameters."""
    payload: dict[str, object] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "outcome": {
            "positive_label": "denial",
            "warning": "Observational post-decision association target; not credit risk.",
        },
        "features": list(FEATURES),
        "standard_scaler": {
            "mean": [float(value) for value in feature_means],
            "scale": [float(value) for value in feature_scales],
        },
        "logistic_regression": {
            "coefficients": [float(value) for value in logistic_coefficients],
            "intercept": float(logistic_intercept),
        },
        "platt_scaler": {
            "coefficient": float(platt_coefficient),
            "intercept": float(platt_intercept),
        },
    }
    # Validate before signing so artifact creation and loading share one strict contract.
    _model_from_payload(payload)
    return {
        **payload,
        "integrity": {"algorithm": "sha256", "payload_sha256": _canonical_sha256(payload)},
    }


def _model_from_payload(payload: Mapping[str, Any]) -> LocalChronologicalModel:
    expected = frozenset(
        {
            "artifact_schema_version",
            "model_version",
            "outcome",
            "features",
            "standard_scaler",
            "logistic_regression",
            "platt_scaler",
        }
    )
    payload = _object(payload, "artifact payload", expected)
    if payload["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactValidationError("unsupported artifact_schema_version")
    if payload["model_version"] != MODEL_VERSION:
        raise ModelArtifactValidationError("unsupported model_version")
    if payload["features"] != list(FEATURES):
        raise ModelArtifactValidationError("features must match the declared inference order")
    outcome = _object(payload["outcome"], "outcome", frozenset({"positive_label", "warning"}))
    if outcome["positive_label"] != "denial" or not isinstance(outcome["warning"], str):
        raise ModelArtifactValidationError("outcome must preserve the denial-association meaning")
    scaler = _object(payload["standard_scaler"], "standard_scaler", frozenset({"mean", "scale"}))
    logistic = _object(
        payload["logistic_regression"],
        "logistic_regression",
        frozenset({"coefficients", "intercept"}),
    )
    platt = _object(
        payload["platt_scaler"], "platt_scaler", frozenset({"coefficient", "intercept"})
    )
    return LocalChronologicalModel(
        feature_means=_finite_vector(scaler["mean"], "standard_scaler.mean"),
        feature_scales=_finite_vector(scaler["scale"], "standard_scaler.scale", positive=True),
        logistic_coefficients=_finite_vector(
            logistic["coefficients"], "logistic_regression.coefficients"
        ),
        logistic_intercept=_finite_number(logistic["intercept"], "logistic_regression.intercept"),
        platt_coefficient=_finite_number(platt["coefficient"], "platt_scaler.coefficient"),
        platt_intercept=_finite_number(platt["intercept"], "platt_scaler.intercept"),
    )


def load_model_artifact(path: str | Path) -> LocalChronologicalModel:
    """Load a checksummed JSON artifact without deserializing executable model formats."""
    try:
        contents = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ModelArtifactValidationError(f"model artifact cannot be read: {path}") from exc
    document = parse_json_object(
        contents, error_type=ModelArtifactValidationError, subject="model artifact"
    )
    expected = frozenset(
        {
            "artifact_schema_version",
            "model_version",
            "outcome",
            "features",
            "standard_scaler",
            "logistic_regression",
            "platt_scaler",
            "integrity",
        }
    )
    document = _object(document, "model artifact", expected)
    integrity = _object(
        document["integrity"], "integrity", frozenset({"algorithm", "payload_sha256"})
    )
    if integrity["algorithm"] != "sha256" or not isinstance(integrity["payload_sha256"], str):
        raise ModelArtifactValidationError("integrity must declare a SHA-256 payload checksum")
    checksum = integrity["payload_sha256"].lower()
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise ModelArtifactValidationError("integrity.payload_sha256 must be a SHA-256 hex digest")
    payload = {key: value for key, value in document.items() if key != "integrity"}
    try:
        payload_checksum = _canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise ModelArtifactValidationError(
            "model artifact payload must contain only finite JSON values"
        ) from exc
    if not hmac.compare_digest(checksum, payload_checksum):
        raise ModelArtifactValidationError("model artifact checksum mismatch")
    return _model_from_payload(payload)
