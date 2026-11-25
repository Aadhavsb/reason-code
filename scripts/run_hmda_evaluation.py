#!/usr/bin/env python3
"""Run the explicit local HMDA chronological model and calibration evaluation.

This research command never downloads data, changes the demo API, or produces lending
recommendations. It fits a simple denial-outcome association model on the manifest's
train partition, fits Platt calibration on validation only, and writes test-only
metrics/artifacts to a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reason_code.data import DatasetValidationError, load_local_dataset
from reason_code.provenance import manifest_hash
from reason_code.research_model import FEATURES, MODEL_VERSION, artifact_document

REVIEW_QUANTILES = (0.4, 0.6)


def _as_arrays(records: tuple[object, ...]) -> tuple[np.ndarray, np.ndarray]:
    rows = [dict(record) for record in records]
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=float)
    # Current public HMDA action codes: 3 is a denial.  The model target is denial,
    # not the loader's generic source label, so this meaning is explicit in the artifact.
    target = np.asarray([int(row["label"] == 0) for row in rows], dtype=int)
    return matrix, target


def _sha256_json(document: object) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def _reliability(y: np.ndarray, probability: np.ndarray) -> list[dict[str, float | int]]:
    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    # calibration_curve does not expose bin sizes, so report the paired estimates only.
    return [
        {"mean_predicted_denial_probability": float(x), "observed_denial_rate": float(z)}
        for x, z in zip(predicted, observed, strict=True)
    ]


def _review_calibration_artifact(
    records: tuple[object, ...], probability: np.ndarray, dataset_manifest_sha256: str
) -> dict[str, object]:
    """Bind held-out calibrated probabilities to an explicit local dataset manifest."""
    if len(records) != len(probability):
        raise ValueError("test records and calibrated probabilities must have the same length")
    predictions = sorted(
        [
            {
                "record_id": str(dict(record)["record_id"]),
                "calibrated_probability": float(score),
            }
            for record, score in zip(records, probability, strict=True)
        ],
        key=lambda item: item["record_id"],
    )
    prediction_bytes = json.dumps(
        predictions, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return {
        "artifact_version": "resume-gate-calibration-v2",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "evaluator_version": MODEL_VERSION,
        "split": "test",
        "predictions": predictions,
        "test_predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
    }


def _abstention(
    y: np.ndarray, probability: np.ndarray, lower_bound: float, upper_bound: float
) -> dict[str, float | int]:
    automated = (probability < lower_bound) | (probability > upper_bound)
    predicted_denial = probability > upper_bound
    accepted = int(automated.sum())
    result: dict[str, float | int] = {
        "review_lower_bound": lower_bound,
        "review_upper_bound": upper_bound,
        "test_rows": int(len(y)),
        "human_review_rows": int((~automated).sum()),
        "automated_rows": accepted,
        "automated_coverage": float(accepted / len(y)),
    }
    if predicted_denial.any():
        result["automated_denial_precision"] = float(
            precision_score(y, predicted_denial, zero_division=0)
        )
        result["automated_denial_recall"] = float(
            recall_score(y, predicted_denial, zero_division=0)
        )
    return result


def run(dataset_manifest: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("output directory must not already exist")
    dataset = load_local_dataset(dataset_manifest)
    train_x, train_y = _as_arrays(dataset.partitions["train"])
    validation_x, validation_y = _as_arrays(dataset.partitions["validation"])
    test_x, test_y = _as_arrays(dataset.partitions["test"])
    if (
        len(np.unique(train_y)) != 2
        or len(np.unique(validation_y)) != 2
        or len(np.unique(test_y)) != 2
    ):
        raise ValueError(
            "every chronological partition must contain originated and denied outcomes"
        )

    base = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1_000, random_state=7)),
        ]
    )
    base.fit(train_x, train_y)
    validation_raw = base.predict_proba(validation_x)[:, 1]
    # Platt calibration is fitted only on validation predictions generated by a train-only model.
    calibrator = LogisticRegression(max_iter=1_000, random_state=7)
    calibrator.fit(validation_raw.reshape(-1, 1), validation_y)
    validation_calibrated = calibrator.predict_proba(validation_raw.reshape(-1, 1))[:, 1]
    review_lower_bound, review_upper_bound = (
        float(value) for value in np.quantile(validation_calibrated, REVIEW_QUANTILES)
    )
    test_raw = base.predict_proba(test_x)[:, 1]
    test_calibrated = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    dataset_manifest_sha256 = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    review_calibration_artifact = _review_calibration_artifact(
        dataset.partitions["test"], test_calibrated, dataset_manifest_sha256
    )
    review_calibration_artifact_sha256 = _sha256_json(review_calibration_artifact)

    model_artifact = artifact_document(
        feature_means=base.named_steps["scale"].mean_,
        feature_scales=base.named_steps["scale"].scale_,
        logistic_coefficients=base.named_steps["logistic"].coef_[0],
        logistic_intercept=base.named_steps["logistic"].intercept_[0],
        platt_coefficient=calibrator.coef_[0, 0],
        platt_intercept=calibrator.intercept_[0],
    )
    model_artifact_sha256 = _sha256_json(model_artifact)

    report: dict[str, object] = {
        "artifact_schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_version": MODEL_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "provenance_manifest_sha256": manifest_hash(dataset.provenance),
        "split": {"train": "2022", "validation": "2023", "test": "2024"},
        "outcome": {
            "positive_label": "denial",
            "source_action_taken_codes": {"originated": "1", "denied": "3"},
            "warning": (
                "This is an observational post-decision association target, not credit risk."
            ),
        },
        "features": list(FEATURES),
        "model_artifact": {
            "path": "model.json",
            "canonical_document_sha256": model_artifact_sha256,
            "warning": (
                "JSON-only research artifact for offline association scoring; it is not used by "
                "the FastAPI decision endpoint."
            ),
        },
        "review_calibration_artifact": {
            "path": "review-calibration.json",
            "canonical_document_sha256": review_calibration_artifact_sha256,
            "warning": (
                "Held-out calibrated probabilities for deterministic proposed review-packet "
                "selection only; not a lending decision policy."
            ),
        },
        "calibration": {
            "method": "Platt scaling (logistic regression) fitted once on validation predictions",
            "validation_raw": _metrics(validation_y, validation_raw),
            "test_raw": _metrics(test_y, test_raw),
            "test_calibrated": _metrics(test_y, test_calibrated),
            "test_reliability_bins": _reliability(test_y, test_calibrated),
        },
        "abstention_diagnostics": {
            "selection": "central 40%–60% validation calibrated-probability quantile band",
            "warning": "Diagnostic only; not a lending decision policy.",
            **_abstention(test_y, test_calibrated, review_lower_bound, review_upper_bound),
        },
        "limitations": [
            "No causal, fairness, compliance, lending-suitability, or loss claim is supported.",
            "Features are limited to income, loan amount, and numeric DTI from the prepared slice.",
            (
                "Chronology is activity-year level because public HMDA data lacks "
                "an event-level date here."
            ),
            "This artifact does not change the FastAPI demo scorer or authorize underwriting use.",
        ],
    }
    output_dir.mkdir(parents=True)
    model_contents = json.dumps(model_artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output_dir / "model.json").write_text(model_contents, encoding="utf-8")
    review_contents = (
        json.dumps(review_calibration_artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output_dir / "review-calibration.json").write_text(review_contents, encoding="utf-8")
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline chronological HMDA evaluation.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = run(arguments.dataset_manifest.expanduser(), arguments.output_dir.expanduser())
    except (DatasetValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
