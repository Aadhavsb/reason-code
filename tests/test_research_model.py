from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reason_code.research_model import (
    ModelArtifactValidationError,
    artifact_document,
    load_model_artifact,
)


def _fitted_models() -> tuple[Pipeline, LogisticRegression, np.ndarray]:
    train_x = np.asarray(
        [
            [60_000, 100_000, 16],
            [70_000, 130_000, 20],
            [85_000, 175_000, 30],
            [95_000, 190_000, 36],
            [110_000, 250_000, 45],
            [125_000, 300_000, 50],
        ],
        dtype=float,
    )
    train_y = np.asarray([0, 0, 0, 1, 1, 1])
    validation_x = np.asarray(
        [
            [75_000, 150_000, 25],
            [100_000, 220_000, 40],
            [130_000, 320_000, 55],
            [65_000, 110_000, 18],
        ],
        dtype=float,
    )
    validation_y = np.asarray([0, 1, 1, 0])
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1_000, random_state=7)),
        ]
    ).fit(train_x, train_y)
    calibrator = LogisticRegression(max_iter=1_000, random_state=7).fit(
        base.predict_proba(validation_x)[:, 1].reshape(-1, 1), validation_y
    )
    return base, calibrator, validation_x


def _document(base: Pipeline, calibrator: LogisticRegression) -> dict[str, object]:
    return artifact_document(
        feature_means=base.named_steps["scale"].mean_,
        feature_scales=base.named_steps["scale"].scale_,
        logistic_coefficients=base.named_steps["logistic"].coef_[0],
        logistic_intercept=base.named_steps["logistic"].intercept_[0],
        platt_coefficient=calibrator.coef_[0, 0],
        platt_intercept=calibrator.intercept_[0],
    )


def test_json_artifact_scorer_matches_fitted_sklearn_pipeline_exactly(tmp_path: Path):
    base, calibrator, rows = _fitted_models()
    artifact_path = tmp_path / "model.json"
    artifact_path.write_text(json.dumps(_document(base, calibrator)), encoding="utf-8")

    scorer = load_model_artifact(artifact_path)
    expected = calibrator.predict_proba(base.predict_proba(rows)[:, 1].reshape(-1, 1))[:, 1]
    actual = np.asarray(
        [
            scorer.score_features(
                {
                    "annual_income": row[0],
                    "requested_amount": row[1],
                    "debt_to_income_ratio": row[2],
                }
            )
            for row in rows
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_loader_rejects_tampering_duplicate_keys_and_nonfinite_values(tmp_path: Path):
    base, calibrator, _ = _fitted_models()
    document = _document(base, calibrator)
    document["platt_scaler"]["intercept"] = 999  # type: ignore[index]
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelArtifactValidationError, match="checksum mismatch"):
        load_model_artifact(altered)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"artifact_schema_version":"1.0","artifact_schema_version":"1.0"}', encoding="utf-8"
    )
    with pytest.raises(ModelArtifactValidationError, match="duplicate JSON key"):
        load_model_artifact(duplicate)

    nonfinite = _document(base, calibrator)
    nonfinite["standard_scaler"]["mean"][0] = float("nan")  # type: ignore[index]
    nonfinite["integrity"]["payload_sha256"] = "0" * 64  # type: ignore[index]
    invalid = tmp_path / "nonfinite.json"
    invalid.write_text(json.dumps(nonfinite, allow_nan=True), encoding="utf-8")
    with pytest.raises(ModelArtifactValidationError, match="only finite JSON values"):
        load_model_artifact(invalid)


def test_offline_evaluation_writes_loadable_json_only_model_artifact(tmp_path: Path, monkeypatch):
    script_path = Path(__file__).parents[1] / "scripts" / "run_hmda_evaluation.py"
    spec = importlib.util.spec_from_file_location("hmda_evaluation", script_path)
    assert spec and spec.loader
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)

    records = {
        "train": tuple(
            {
                "annual_income": income,
                "requested_amount": amount,
                "debt_to_income_ratio": dti,
                "label": label,
            }
            for income, amount, dti, label in [
                (60_000, 100_000, 16, 1),
                (70_000, 130_000, 20, 1),
                (95_000, 190_000, 36, 0),
                (125_000, 300_000, 50, 0),
            ]
        ),
        "validation": tuple(
            {
                "annual_income": income,
                "requested_amount": amount,
                "debt_to_income_ratio": dti,
                "label": label,
            }
            for income, amount, dti, label in [
                (75_000, 150_000, 25, 1),
                (65_000, 110_000, 18, 1),
                (100_000, 220_000, 40, 0),
                (130_000, 320_000, 55, 0),
            ]
        ),
        "test": tuple(
            {
                "annual_income": income,
                "requested_amount": amount,
                "debt_to_income_ratio": dti,
                "label": label,
            }
            for income, amount, dti, label in [
                (80_000, 160_000, 26, 1),
                (68_000, 120_000, 19, 1),
                (105_000, 230_000, 42, 0),
                (135_000, 330_000, 56, 0),
            ]
        ),
    }

    for partition, partition_records in records.items():
        for index, record in enumerate(partition_records):
            record["record_id"] = f"{partition}-{index}"

    class Dataset:
        partitions = records
        provenance = {
            "schema_version": "1.0",
            "source_url": "https://example.test/source.csv",
            "filter": {},
            "download_date": "2026-01-01",
            "fields": ["income", "loan_amount", "debt_to_income_ratio"],
            "source_version_hash": "a" * 64,
        }

    monkeypatch.setattr(evaluation, "load_local_dataset", lambda _path: Dataset())
    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "output"

    report = evaluation.run(manifest, output_dir)

    artifact_path = output_dir / "model.json"
    review_calibration_path = output_dir / "review-calibration.json"
    assert artifact_path.exists()
    assert review_calibration_path.exists()
    assert report["model_artifact"]["path"] == "model.json"  # type: ignore[index]
    assert report["review_calibration_artifact"]["path"] == "review-calibration.json"  # type: ignore[index]
    assert (
        load_model_artifact(artifact_path).score_features(
            {"annual_income": 80_000, "requested_amount": 160_000, "debt_to_income_ratio": 26}
        )
        > 0
    )
