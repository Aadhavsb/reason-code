from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from reason_code.evaluation.human_review import (
    ANNOTATION_VERSION,
    HumanReviewError,
    PacketSelection,
    StratumRequest,
    export_review_artifacts,
    freeze_review,
    main,
    review_report,
    select_resume_gate_packet,
    select_review_packet,
    validate_adjudications,
    validate_annotation,
    validate_independent_reviews,
)


def _manifest(tmp_path: Path, *, test_rows: int = 4) -> Path:
    rows = [
        "loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio",
        "train-source,2023-06-01,1,70000,180000,28",
        "validation-source,2024-06-01,3,75000,190000,29",
    ]
    for index in range(test_rows):
        label = "1" if index % 2 == 0 else "3"
        rows.append(f"source-{index},2025-06-01,{label},{80000 + index},200000,30")
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_url": "https://example.gov/synthetic.csv",
                "filter": {"synthetic": True},
                "download_date": "2026-11-23",
                "fields": [
                    "loan_id",
                    "activity_date",
                    "action_taken",
                    "income",
                    "loan_amount",
                    "debt_to_income_ratio",
                ],
                "source_version_hash": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_path": "dataset.csv",
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "provenance_manifest_path": "provenance.json",
                "record_id_column": "loan_id",
                "record_date_column": "activity_date",
                "target_column": "action_taken",
                "target_mapping": {"1": 1, "3": 0},
                "feature_columns": {
                    "annual_income": "income",
                    "requested_amount": "loan_amount",
                    "debt_to_income_ratio": "debt_to_income_ratio",
                },
                "split": {"train_end": "2023-12-31", "validation_end": "2024-12-31"},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _resume_artifact(tmp_path: Path, manifest: Path, *, test_rows: int = 75) -> Path:
    predictions = []
    for index in range(test_rows):
        predictions.append(
            {
                "record_id": f"source-{index}",
                "calibrated_probability": 0.05 + (index * 0.0003),
            }
        )
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_version": "resume-gate-calibration-v2",
                "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "evaluator_version": "synthetic-evaluator-v1",
                "split": "test",
                "predictions": predictions,
                "test_predictions_sha256": hashlib.sha256(
                    json.dumps(
                        sorted(predictions, key=lambda item: item["record_id"]),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("ascii")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _selection() -> PacketSelection:
    return PacketSelection(
        partition="test",
        seed="synthetic-test-seed",
        strata=(
            StratumRequest("approved", 1, natural_count=1, controlled_negative_count=1),
            StratumRequest("denied", 0, natural_count=1, controlled_negative_count=1),
        ),
    )


def _annotations(packet: dict[str, object]) -> list[dict[str, str]]:
    annotations = []
    for reviewer, verdicts in (
        ("reviewer-a", ("accept", "reject", "accept", "reject")),
        ("reviewer-b", ("accept", "accept", "accept", "reject")),
    ):
        for case, verdict in zip(packet["cases"], verdicts, strict=True):
            annotations.append(
                {
                    "annotation_version": ANNOTATION_VERSION,
                    "packet_id": str(packet["packet_id"]),
                    "reviewer_id": reviewer,
                    "case_id": str(case["case_id"]),
                    "verdict": verdict,
                    "notes": "reviewed against displayed facts",
                }
            )
    return annotations


def test_builds_fixed_gate_from_compressed_empirical_predictions(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, test_rows=75)
    artifact = _resume_artifact(tmp_path, manifest)
    assert (
        max(
            item["calibrated_probability"]
            for item in json.loads(artifact.read_text())["predictions"]
        )
        < 0.075
    )
    export = select_resume_gate_packet(manifest, artifact)

    assert export.packet["case_count"] == 75
    serialized_packet = json.dumps(export.packet)
    for forbidden in ("source-", "calibrated_probability", "observed_outcome", "case_type"):
        assert forbidden not in serialized_packet
    case_types = [item["case_type"] for item in export.private_linkage["cases"]]
    assert Counter(case_types) == {
        "natural": 50,
        "controlled_invalid": 16,
        "controlled_valid": 9,
    }
    natural = [item for item in export.private_linkage["cases"] if item["case_type"] == "natural"]
    assert Counter((item["observed_outcome"], item["probability_band"]) for item in natural) == {
        (label, band): 5 for label in (0, 1) for band in ("q1", "q2", "q3", "q4", "q5")
    }
    invalid = [
        item
        for item in export.private_linkage["cases"]
        if item["case_type"] == "controlled_invalid"
    ]
    assert Counter(item["visible_contradiction_type"] for item in invalid) == {
        "annual_income_plus_one": 2,
        "requested_amount_plus_one": 2,
        "debt_to_income_plus_one": 2,
        "annual_income_as_requested_amount": 2,
        "requested_amount_as_annual_income": 2,
        "debt_to_income_as_annual_income": 2,
        "annual_income_as_debt_to_income": 2,
        "requested_amount_as_debt_to_income": 2,
    }
    assert len({item["source_record_id"] for item in export.private_linkage["cases"]}) == 75

    public_by_id = {item["case_id"]: item for item in export.packet["cases"]}
    visible_verdicts = {}
    for linkage in export.private_linkage["cases"]:
        case = public_by_id[linkage["case_id"]]
        visible_key = json.dumps(
            {"facts": case["facts"], "candidate_statement": case["candidate_statement"]},
            sort_keys=True,
        )
        visible_verdicts.setdefault(visible_key, linkage["expected_verdict"])
        assert visible_verdicts[visible_key] == linkage["expected_verdict"]
    for linkage in invalid:
        case = public_by_id[linkage["case_id"]]
        statement = case["candidate_statement"]
        number = float(re.search(r"([0-9]+\.[0-9]+)", statement).group(1))
        if statement.startswith("Annual income"):
            assert number != case["facts"]["annual_income"]
        elif statement.startswith("Requested amount"):
            assert number != case["facts"]["requested_amount"]
        else:
            assert number != case["facts"]["debt_to_income_ratio"]
    assert (
        len(
            {
                json.dumps(
                    {
                        "facts": public_by_id[item["case_id"]]["facts"],
                        "candidate_statement": public_by_id[item["case_id"]]["candidate_statement"],
                    },
                    sort_keys=True,
                )
                for item in invalid
            }
        )
        == 16
    )


def test_resume_gate_fails_closed_on_quantile_stratum_shortage(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, test_rows=40)
    with pytest.raises(HumanReviewError, match="observed_label_0_q1"):
        select_resume_gate_packet(manifest, _resume_artifact(tmp_path, manifest, test_rows=40))


def test_resume_gate_cli_exports_separate_local_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _manifest(tmp_path, test_rows=75)
    artifact = _resume_artifact(tmp_path, manifest)
    packet = tmp_path / "packet.json"
    linkage = tmp_path / "linkage.json"

    assert (
        main(
            [
                "--dataset-manifest",
                str(manifest),
                "--model-artifact",
                str(artifact),
                "--packet",
                str(packet),
                "--private-linkage",
                str(linkage),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["case_count"] == 75
    assert json.loads(packet.read_text(encoding="utf-8"))["case_count"] == 75
    assert "source-" not in packet.read_text(encoding="utf-8")
    assert "source-" in linkage.read_text(encoding="utf-8")


def test_selects_deterministic_blinded_packet_and_keeps_linkage_private(tmp_path: Path) -> None:
    export = select_review_packet(_manifest(tmp_path), _selection())

    assert export.packet["case_count"] == 4
    serialized_packet = json.dumps(export.packet)
    assert "source-" not in serialized_packet
    assert "case_type" not in serialized_packet
    assert "expected_verdict" not in serialized_packet
    assert {item["case_type"] for item in export.private_linkage["cases"]} == {
        "natural",
        "controlled_negative",
    }
    assert {item["expected_verdict"] for item in export.private_linkage["cases"]} == {
        "accept",
        "reject",
    }

    packet_path = tmp_path / "packet.json"
    linkage_path = tmp_path / "private-linkage.json"
    export_review_artifacts(export, packet_path, linkage_path)
    assert "source-" not in packet_path.read_text(encoding="utf-8")
    assert "source-" in linkage_path.read_text(encoding="utf-8")
    with pytest.raises(HumanReviewError, match="paths must differ"):
        export_review_artifacts(export, packet_path, packet_path)


def test_fails_closed_when_any_requested_stratum_is_short(tmp_path: Path) -> None:
    with pytest.raises(HumanReviewError, match="stratum shortage for approved"):
        select_review_packet(_manifest(tmp_path, test_rows=2), _selection())


def test_validates_independent_reviews_adjudication_freeze_and_report(tmp_path: Path) -> None:
    export = select_review_packet(_manifest(tmp_path), _selection())
    packet = dict(export.packet)
    annotations = _annotations(packet)
    reviews = validate_independent_reviews(packet, annotations)
    disagreements = [
        {
            "packet_id": packet["packet_id"],
            "case_id": packet["cases"][1]["case_id"],
            "verdict": "reject",
            "rationale": "The candidate conflicts with the displayed income.",
        }
    ]
    adjudications = validate_adjudications(packet, reviews, disagreements, "adjudicator")
    freeze = freeze_review(packet, reviews, adjudications, "adjudicator")
    report = review_report(packet, export.private_linkage, freeze)

    assert freeze["case_count"] == 4
    assert len(freeze["packet_sha256"]) == 64
    assert report["controlled_case_counts"] == {"controlled_negative": 2, "natural": 2}

    tampered_packet = json.loads(json.dumps(packet))
    tampered_packet["cases"][0]["candidate_statement"] = "changed after freeze"
    with pytest.raises(HumanReviewError, match="freeze certificate does not match"):
        review_report(tampered_packet, export.private_linkage, freeze)


def test_annotation_and_adjudication_fail_on_packet_or_review_boundary(tmp_path: Path) -> None:
    export = select_review_packet(_manifest(tmp_path), _selection())
    packet = dict(export.packet)
    annotation = _annotations(packet)[0]
    annotation["case_id"] = "source-0"
    with pytest.raises(HumanReviewError, match="invalid case"):
        validate_annotation(annotation, packet)

    reviews = validate_independent_reviews(packet, _annotations(packet))
    with pytest.raises(HumanReviewError, match="adjudicator must be independent"):
        validate_adjudications(packet, reviews, [], "reviewer-a")
    with pytest.raises(HumanReviewError, match="adjudications must cover"):
        validate_adjudications(packet, reviews, [], "adjudicator")
