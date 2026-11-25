"""Deterministic, test-only human-review packet utilities.

This module deliberately consumes only a caller-supplied local dataset manifest.  It
never loads a model, calls a service, or imports the demo API.  Public packets contain
opaque case identifiers; the source-record linkage is a separate private artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from reason_code.data import DatasetValidationError, load_local_dataset
from reason_code.provenance import parse_json_object

PACKET_VERSION = "human-review-packet-v1"
ANNOTATION_VERSION = "human-review-annotation-v1"
FREEZE_VERSION = "human-review-freeze-v1"
_VALID_VERDICTS = frozenset({"accept", "reject", "uncertain"})


class HumanReviewError(ValueError):
    """Raised when a review artifact is incomplete, linked, or non-reproducible."""


def _canonical_bytes(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _digest(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _opaque_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _require_exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise HumanReviewError(f"invalid {label} schema")
    return value


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise HumanReviewError(f"{label} must be a non-empty single-line string")
    return value.strip()


def _reviewer_id(value: Any) -> str:
    identifier = _non_empty_string(value, "reviewer_id")
    if len(identifier) > 128:
        raise HumanReviewError("reviewer_id is too long")
    return identifier


@dataclass(frozen=True)
class StratumRequest:
    """Requested natural and controlled-negative cases for one binary label stratum."""

    name: str
    label: int
    natural_count: int
    controlled_negative_count: int

    def __post_init__(self) -> None:
        if not self.name or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in self.name
        ):
            raise HumanReviewError("stratum name must be lowercase ASCII")
        if type(self.label) is not int or self.label not in {0, 1}:
            raise HumanReviewError("stratum label must be 0 or 1")
        for value in (self.natural_count, self.controlled_negative_count):
            if type(value) is not int or value < 1:
                raise HumanReviewError("each stratum count must be a positive integer")


@dataclass(frozen=True)
class PacketSelection:
    """Explicit, deterministic sampling instructions for one local dataset partition."""

    partition: Literal["validation", "test"]
    strata: tuple[StratumRequest, ...]
    seed: str

    def __post_init__(self) -> None:
        if self.partition not in {"validation", "test"}:
            raise HumanReviewError("packet selection partition must be validation or test")
        if not self.strata or len({item.name for item in self.strata}) != len(self.strata):
            raise HumanReviewError("packet selection requires uniquely named strata")
        if not self.seed or len(self.seed) > 256:
            raise HumanReviewError("packet selection requires a bounded non-empty seed")


@dataclass(frozen=True)
class ReviewExport:
    """Separate public packet and private linkage; callers must store them separately."""

    packet: Mapping[str, Any]
    private_linkage: Mapping[str, Any]


def _statement(record: Mapping[str, Any], *, altered: bool) -> str:
    income = float(record["annual_income"])
    if altered:
        income += 1.0
    return f"Annual income reported: {income:.2f}."


def _facts(record: Mapping[str, Any]) -> dict[str, float]:
    return {
        "annual_income": float(record["annual_income"]),
        "requested_amount": float(record["requested_amount"]),
        "debt_to_income_ratio": float(record["debt_to_income_ratio"]),
    }


def _manifest_fingerprint(dataset_manifest_path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(dataset_manifest_path).read_bytes()).hexdigest()
    except OSError as exc:
        raise HumanReviewError("dataset manifest cannot be read for packet freeze") from exc


def select_review_packet(
    dataset_manifest_path: str | Path, selection: PacketSelection
) -> ReviewExport:
    """Select exact requested strata, fail on shortage, and make a blinded packet.

    Natural cases have a candidate consistent with shown facts. Controlled negatives use
    the same source-record pool but contain one deterministic, visible contradiction.
    Neither the packet nor its IDs expose source record IDs, labels, split, or case type.
    """
    try:
        dataset = load_local_dataset(dataset_manifest_path)
    except DatasetValidationError as exc:
        raise HumanReviewError(f"invalid local dataset manifest: {exc}") from exc
    manifest_hash = _manifest_fingerprint(dataset_manifest_path)
    selected: list[tuple[StratumRequest, Mapping[str, Any], str]] = []
    for stratum in selection.strata:
        pool = [
            record
            for record in dataset.partitions[selection.partition]
            if record["label"] == stratum.label
        ]
        ranked = sorted(
            pool,
            key=lambda record: _opaque_id(
                manifest_hash, selection.seed, stratum.name, str(record["record_id"])
            ),
        )
        needed = stratum.natural_count + stratum.controlled_negative_count
        if len(ranked) < needed:
            raise HumanReviewError(
                f"stratum shortage for {stratum.name}: requested {needed}, available {len(ranked)}"
            )
        selected.extend((stratum, record, "natural") for record in ranked[: stratum.natural_count])
        selected.extend(
            (stratum, record, "controlled_negative")
            for record in ranked[stratum.natural_count : needed]
        )

    packet_cases: list[dict[str, Any]] = []
    linkage_cases: list[dict[str, Any]] = []
    for stratum, record, case_type in selected:
        case_id = _opaque_id(
            manifest_hash, selection.seed, stratum.name, str(record["record_id"]), case_type
        )
        packet_cases.append(
            {
                "case_id": case_id,
                "facts": _facts(record),
                "candidate_statement": _statement(
                    record, altered=case_type == "controlled_negative"
                ),
            }
        )
        linkage_cases.append(
            {
                "case_id": case_id,
                "source_record_id": str(record["record_id"]),
                "stratum": stratum.name,
                "case_type": case_type,
                "expected_verdict": "accept" if case_type == "natural" else "reject",
            }
        )
    packet_cases.sort(
        key=lambda item: _opaque_id(manifest_hash, selection.seed, item["case_id"], "order")
    )
    packet_id = _opaque_id(
        manifest_hash, selection.seed, "packet", *sorted(item["case_id"] for item in packet_cases)
    )
    packet: dict[str, Any] = {
        "packet_version": PACKET_VERSION,
        "packet_id": packet_id,
        "case_count": len(packet_cases),
        "cases": packet_cases,
    }
    linkage: dict[str, Any] = {
        "linkage_version": PACKET_VERSION,
        "packet_id": packet_id,
        "dataset_manifest_sha256": manifest_hash,
        "case_count": len(linkage_cases),
        "cases": sorted(linkage_cases, key=lambda item: item["case_id"]),
    }
    _validate_packet(packet)
    return ReviewExport(packet=packet, private_linkage=linkage)


def export_review_artifacts(
    export: ReviewExport, packet_path: str | Path, linkage_path: str | Path
) -> None:
    """Write public and private JSON to distinct paths, refusing co-location."""
    packet_target = Path(packet_path).resolve()
    linkage_target = Path(linkage_path).resolve()
    if packet_target == linkage_target:
        raise HumanReviewError("public packet and private linkage paths must differ")
    _validate_packet(export.packet)
    if export.private_linkage.get("packet_id") != export.packet["packet_id"]:
        raise HumanReviewError("private linkage does not match packet")
    packet_target.write_bytes(_canonical_bytes(export.packet) + b"\n")
    linkage_target.write_bytes(_canonical_bytes(export.private_linkage) + b"\n")


def _validate_packet(packet: Mapping[str, Any]) -> None:
    raw = _require_exact_keys(
        packet, {"packet_version", "packet_id", "case_count", "cases"}, "packet"
    )
    if raw["packet_version"] != PACKET_VERSION or not isinstance(raw["packet_id"], str):
        raise HumanReviewError("invalid packet metadata")
    if (
        type(raw["case_count"]) is not int
        or not isinstance(raw["cases"], list)
        or raw["case_count"] != len(raw["cases"])
    ):
        raise HumanReviewError("invalid packet case count")
    identifiers: set[str] = set()
    for case in raw["cases"]:
        _require_exact_keys(case, {"case_id", "facts", "candidate_statement"}, "packet case")
        case_id = _non_empty_string(case["case_id"], "case_id")
        if case_id in identifiers or not isinstance(case["facts"], Mapping):
            raise HumanReviewError("invalid packet case")
        identifiers.add(case_id)


def validate_annotation(annotation: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one blinded review annotation without consulting private linkage."""
    _validate_packet(packet)
    raw = _require_exact_keys(
        annotation,
        {"annotation_version", "packet_id", "reviewer_id", "case_id", "verdict", "notes"},
        "annotation",
    )
    if raw["annotation_version"] != ANNOTATION_VERSION or raw["packet_id"] != packet["packet_id"]:
        raise HumanReviewError("annotation does not match packet")
    case_ids = {case["case_id"] for case in packet["cases"]}
    if raw["case_id"] not in case_ids or raw["verdict"] not in _VALID_VERDICTS:
        raise HumanReviewError("annotation has invalid case or verdict")
    normalized = {
        "annotation_version": ANNOTATION_VERSION,
        "packet_id": packet["packet_id"],
        "reviewer_id": _reviewer_id(raw["reviewer_id"]),
        "case_id": raw["case_id"],
        "verdict": raw["verdict"],
        "notes": _non_empty_string(raw["notes"], "notes"),
    }
    return normalized


def validate_independent_reviews(
    packet: Mapping[str, Any], annotations: Sequence[Mapping[str, Any]], reviewer_count: int = 2
) -> tuple[dict[str, Any], ...]:
    """Require exactly one complete blind annotation set per independent reviewer."""
    _validate_packet(packet)
    if type(reviewer_count) is not int or reviewer_count < 2:
        raise HumanReviewError("independent review requires at least two reviewers")
    validated = tuple(validate_annotation(annotation, packet) for annotation in annotations)
    by_reviewer: dict[str, list[dict[str, Any]]] = {}
    for annotation in validated:
        by_reviewer.setdefault(annotation["reviewer_id"], []).append(annotation)
    if len(by_reviewer) != reviewer_count:
        raise HumanReviewError("independent review has the wrong number of reviewers")
    case_ids = {case["case_id"] for case in packet["cases"]}
    for reviewer, entries in by_reviewer.items():
        if {entry["case_id"] for entry in entries} != case_ids or len(entries) != len(case_ids):
            raise HumanReviewError(f"reviewer {reviewer} must annotate every case exactly once")
    return tuple(sorted(validated, key=lambda item: (item["reviewer_id"], item["case_id"])))


def validate_adjudications(
    packet: Mapping[str, Any],
    reviews: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    adjudicator_id: str,
) -> tuple[dict[str, Any], ...]:
    """Require adjudication exactly for reviewer disagreements and never by a reviewer."""
    validated_reviews = validate_independent_reviews(packet, reviews)
    adjudicator = _reviewer_id(adjudicator_id)
    reviewers = {item["reviewer_id"] for item in validated_reviews}
    if adjudicator in reviewers:
        raise HumanReviewError("adjudicator must be independent of reviewers")
    votes: dict[str, set[str]] = {}
    for review in validated_reviews:
        votes.setdefault(review["case_id"], set()).add(review["verdict"])
    disagreements = {case_id for case_id, verdicts in votes.items() if len(verdicts) > 1}
    normalized: list[dict[str, Any]] = []
    for adjudication in adjudications:
        raw = _require_exact_keys(
            adjudication, {"packet_id", "case_id", "verdict", "rationale"}, "adjudication"
        )
        if (
            raw["packet_id"] != packet["packet_id"]
            or raw["case_id"] not in disagreements
            or raw["verdict"] not in _VALID_VERDICTS
        ):
            raise HumanReviewError("adjudication has invalid packet, case, or verdict")
        normalized.append(
            {
                "packet_id": packet["packet_id"],
                "case_id": raw["case_id"],
                "verdict": raw["verdict"],
                "rationale": _non_empty_string(raw["rationale"], "rationale"),
            }
        )
    if {item["case_id"] for item in normalized} != disagreements or len(normalized) != len(
        disagreements
    ):
        raise HumanReviewError("adjudications must cover each disagreement exactly once")
    return tuple(sorted(normalized, key=lambda item: item["case_id"]))


def freeze_review(
    packet: Mapping[str, Any],
    reviews: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    adjudicator_id: str,
) -> dict[str, Any]:
    """Create a reproducible freeze certificate after all review validation passes."""
    validated_reviews = validate_independent_reviews(packet, reviews)
    validated_adjudications = validate_adjudications(
        packet, validated_reviews, adjudications, adjudicator_id
    )
    votes: dict[str, list[str]] = {}
    for review in validated_reviews:
        votes.setdefault(review["case_id"], []).append(review["verdict"])
    final: dict[str, str] = {}
    adjudicated = {item["case_id"]: item["verdict"] for item in validated_adjudications}
    for case_id, case_votes in votes.items():
        final[case_id] = adjudicated.get(case_id, case_votes[0])
    certificate = {
        "freeze_version": FREEZE_VERSION,
        "packet_id": packet["packet_id"],
        "packet_sha256": _digest(packet),
        "reviews_sha256": _digest(list(validated_reviews)),
        "adjudications_sha256": _digest(list(validated_adjudications)),
        "case_count": len(packet["cases"]),
        "reviewer_count": len({item["reviewer_id"] for item in validated_reviews}),
        "adjudication_count": len(validated_adjudications),
        "final_verdict_counts": dict(sorted(Counter(final.values()).items())),
    }
    return certificate


def review_report(
    packet: Mapping[str, Any], linkage: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
    """Produce an aggregate post-freeze report; linkage never enters the public packet."""
    _validate_packet(packet)
    required = {
        "freeze_version",
        "packet_id",
        "packet_sha256",
        "case_count",
        "final_verdict_counts",
    }
    if (
        not isinstance(freeze, Mapping)
        or not required.issubset(freeze)
        or freeze["freeze_version"] != FREEZE_VERSION
    ):
        raise HumanReviewError("invalid freeze certificate")
    if freeze["packet_id"] != packet["packet_id"] or freeze["packet_sha256"] != _digest(packet):
        raise HumanReviewError("freeze certificate does not match packet")
    if (
        linkage.get("packet_id") != packet["packet_id"]
        or linkage.get("case_count") != packet["case_count"]
    ):
        raise HumanReviewError("private linkage does not match packet")
    cases = linkage.get("cases")
    if not isinstance(cases, list) or len(cases) != packet["case_count"]:
        raise HumanReviewError("invalid private linkage")
    controls = Counter(item.get("case_type") for item in cases)
    allowed_case_types = {
        "natural",
        "controlled_negative",
        "controlled_invalid",
        "controlled_valid",
    }
    if not set(controls).issubset(allowed_case_types):
        raise HumanReviewError("private linkage has invalid case types")
    return {
        "report_version": FREEZE_VERSION,
        "packet_id": packet["packet_id"],
        "packet_sha256": freeze["packet_sha256"],
        "case_count": freeze["case_count"],
        "controlled_case_counts": dict(sorted(controls.items())),
        "final_verdict_counts": freeze["final_verdict_counts"],
    }


RESUME_GATE_PACKET_SIZE = 75
RESUME_GATE_SEED = "resume-gate-proposed-v1"
_QUANTILE_BANDS = ("q1", "q2", "q3", "q4", "q5")
_VISIBLE_CONTRADICTION_TYPES = (
    "annual_income_plus_one",
    "requested_amount_plus_one",
    "debt_to_income_plus_one",
    "annual_income_as_requested_amount",
    "requested_amount_as_annual_income",
    "debt_to_income_as_annual_income",
    "annual_income_as_debt_to_income",
    "requested_amount_as_debt_to_income",
)


def _artifact_fingerprint(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise HumanReviewError("model artifact cannot be read") from exc


def _load_calibrated_probabilities(
    artifact_path: str | Path, manifest_hash: str, test_ids: set[str]
) -> tuple[dict[str, float], str]:
    """Load only an explicit local JSON calibration artifact, with no inference."""
    try:
        payload = Path(artifact_path).read_text(encoding="utf-8")
        raw = parse_json_object(payload, error_type=HumanReviewError, subject="model artifact")
    except (OSError, UnicodeError, HumanReviewError) as exc:
        raise HumanReviewError("model artifact cannot be read as strict JSON") from exc
    _require_exact_keys(
        raw,
        {
            "artifact_version",
            "dataset_manifest_sha256",
            "evaluator_version",
            "split",
            "predictions",
            "test_predictions_sha256",
        },
        "model artifact",
    )
    if raw["artifact_version"] != "resume-gate-calibration-v2":
        raise HumanReviewError("unsupported model artifact version")
    if (
        raw["dataset_manifest_sha256"] != manifest_hash
        or raw["split"] != "test"
        or not isinstance(raw["predictions"], list)
        or not isinstance(raw["evaluator_version"], str)
        or not raw["evaluator_version"].strip()
    ):
        raise HumanReviewError("model artifact does not bind to the supplied test evaluation")
    probabilities: dict[str, float] = {}
    for prediction in raw["predictions"]:
        _require_exact_keys(prediction, {"record_id", "calibrated_probability"}, "model prediction")
        record_id = _non_empty_string(prediction["record_id"], "model prediction record_id")
        probability = prediction["calibrated_probability"]
        if (
            type(probability) not in {int, float}
            or not math.isfinite(float(probability))
            or not 0 <= float(probability) <= 1
            or record_id in probabilities
        ):
            raise HumanReviewError("model artifact has invalid calibrated probabilities")
        probabilities[record_id] = float(probability)
    if set(probabilities) != test_ids:
        raise HumanReviewError(
            "model artifact predictions must match held-out test records exactly"
        )
    canonical_predictions = [
        {"record_id": record_id, "calibrated_probability": probabilities[record_id]}
        for record_id in sorted(probabilities)
    ]
    if raw["test_predictions_sha256"] != _digest(canonical_predictions):
        raise HumanReviewError("model artifact test prediction hash does not match predictions")
    return probabilities, _artifact_fingerprint(artifact_path)


def _empirical_quantile_bands(
    probabilities: Mapping[str, float], manifest_hash: str, artifact_hash: str
) -> dict[str, str]:
    """Assign five deterministic rank bands from the bound held-out predictions.

    Equal probabilities are resolved only by a hash of bound artifacts and record ID,
    so compressed or tied calibrated ranges never collapse into one numeric band.
    """
    if len(probabilities) < 5:
        raise HumanReviewError(
            "held-out predictions require at least five records for quantile bands"
        )
    ordered_ids = sorted(
        probabilities,
        key=lambda record_id: (
            probabilities[record_id],
            _opaque_id(manifest_hash, artifact_hash, "quantile-tie", record_id),
        ),
    )
    return {
        record_id: _QUANTILE_BANDS[min(4, index * 5 // len(ordered_ids))]
        for index, record_id in enumerate(ordered_ids)
    }


def _resume_case(
    case_id: str, record: Mapping[str, Any], candidate_statement: str
) -> dict[str, Any]:
    return {"case_id": case_id, "facts": _facts(record), "candidate_statement": candidate_statement}


def _invalid_control_statement(control_type: str, record: Mapping[str, Any]) -> str:
    """Create a distinct contradiction that reviewers can verify from displayed facts."""
    income = float(record["annual_income"])
    amount = float(record["requested_amount"])
    debt_to_income = float(record["debt_to_income_ratio"])
    variants = {
        "annual_income_plus_one": f"Annual income reported: {income + 1:.2f}.",
        "requested_amount_plus_one": f"Requested amount reported: {amount + 1:.2f}.",
        "debt_to_income_plus_one": f"Debt-to-income ratio reported: {debt_to_income + 1:.2f}%.",
        "annual_income_as_requested_amount": f"Annual income reported: {amount:.2f}.",
        "requested_amount_as_annual_income": f"Requested amount reported: {income:.2f}.",
        "debt_to_income_as_annual_income": f"Debt-to-income ratio reported: {income:.2f}%.",
        "annual_income_as_debt_to_income": f"Annual income reported: {debt_to_income:.2f}.",
        "requested_amount_as_debt_to_income": f"Requested amount reported: {debt_to_income:.2f}.",
    }
    return variants[control_type]


def select_resume_gate_packet(
    dataset_manifest_path: str | Path,
    model_artifact_path: str | Path,
    *,
    seed: str = RESUME_GATE_SEED,
) -> ReviewExport:
    """Build the fixed proposed 75-case resume gate from explicit local inputs.

    The 50 natural cases are five records from every observed-outcome/empirical-
    quantile stratum. The 25 blinded controls comprise sixteen visibly contradictory
    candidates (two per contradiction type) and nine valid candidates (three per q1/q3/q5
    rank region). Every selection is independently hash ranked using the
    manifest/artifact fingerprints, declared context, and seed.
    """
    if not seed or len(seed) > 256:
        raise HumanReviewError("resume gate requires a bounded non-empty seed")
    try:
        dataset = load_local_dataset(dataset_manifest_path)
    except DatasetValidationError as exc:
        raise HumanReviewError(f"invalid local dataset manifest: {exc}") from exc
    manifest_hash = _manifest_fingerprint(dataset_manifest_path)
    test_records = tuple(dataset.partitions["test"])
    records_by_id = {str(record["record_id"]): record for record in test_records}
    probabilities, artifact_hash = _load_calibrated_probabilities(
        model_artifact_path, manifest_hash, set(records_by_id)
    )
    quantile_bands = _empirical_quantile_bands(probabilities, manifest_hash, artifact_hash)
    natural: list[tuple[Mapping[str, Any], int, str]] = []
    for label in (0, 1):
        for band in _QUANTILE_BANDS:
            pool = [
                record
                for record in test_records
                if record["label"] == label and quantile_bands[str(record["record_id"])] == band
            ]
            ranked = sorted(
                pool,
                key=lambda record: _opaque_id(
                    manifest_hash,
                    artifact_hash,
                    seed,
                    "natural",
                    str(label),
                    band,
                    str(record["record_id"]),
                ),
            )
            if len(ranked) < 5:
                raise HumanReviewError(
                    f"stratum shortage for observed_label_{label}_{band}: "
                    f"requested 5, available {len(ranked)}"
                )
            natural.extend((record, label, band) for record in ranked[:5])

    packet_cases: list[dict[str, Any]] = []
    linkage_cases: list[dict[str, Any]] = []
    for record, label, band in natural:
        record_id = str(record["record_id"])
        case_id = _opaque_id(manifest_hash, artifact_hash, seed, "natural", record_id)
        packet_cases.append(_resume_case(case_id, record, _statement(record, altered=False)))
        linkage_cases.append(
            {
                "case_id": case_id,
                "source_record_id": record_id,
                "case_type": "natural",
                "observed_outcome": label,
                "probability_band": band,
                "calibrated_probability": probabilities[record_id],
                "expected_verdict": "accept",
            }
        )

    consumed_ids = {str(record["record_id"]) for record, _, _ in natural}
    valid_controls: list[tuple[str, Mapping[str, Any]]] = []
    for region in ("q1", "q3", "q5"):
        pool = [
            record
            for record in test_records
            if str(record["record_id"]) not in consumed_ids
            and quantile_bands[str(record["record_id"])] == region
        ]
        ranked = sorted(
            pool,
            key=lambda record: _opaque_id(
                manifest_hash,
                artifact_hash,
                seed,
                "valid-control",
                region,
                str(record["record_id"]),
            ),
        )
        if len(ranked) < 3:
            raise HumanReviewError(
                f"stratum shortage for valid_control_{region}: requested 3, available {len(ranked)}"
            )
        for record in ranked[:3]:
            consumed_ids.add(str(record["record_id"]))
            valid_controls.append((region, record))

    for control_type in _VISIBLE_CONTRADICTION_TYPES:
        pool = [record for record in test_records if str(record["record_id"]) not in consumed_ids]
        ranked = sorted(
            pool,
            key=lambda record: _opaque_id(
                manifest_hash,
                artifact_hash,
                seed,
                "visible-invalid-control",
                control_type,
                str(record["record_id"]),
            ),
        )
        if len(ranked) < 2:
            raise HumanReviewError(
                f"stratum shortage for invalid_control_{control_type}: "
                f"requested 2, available {len(ranked)}"
            )
        for index, record in enumerate(ranked[:2]):
            record_id = str(record["record_id"])
            consumed_ids.add(record_id)
            case_id = _opaque_id(
                manifest_hash, artifact_hash, seed, "invalid", control_type, str(index)
            )
            packet_cases.append(
                _resume_case(case_id, record, _invalid_control_statement(control_type, record))
            )
            linkage_cases.append(
                {
                    "case_id": case_id,
                    "source_record_id": record_id,
                    "case_type": "controlled_invalid",
                    "visible_contradiction_type": control_type,
                    "expected_verdict": "reject",
                }
            )

    for region, record in valid_controls:
        record_id = str(record["record_id"])
        case_id = _opaque_id(manifest_hash, artifact_hash, seed, "valid", region, record_id)
        packet_cases.append(_resume_case(case_id, record, _statement(record, altered=False)))
        linkage_cases.append(
            {
                "case_id": case_id,
                "source_record_id": record_id,
                "case_type": "controlled_valid",
                "probability_region": region,
                "expected_verdict": "accept",
            }
        )

    source_ids = [item["source_record_id"] for item in linkage_cases]
    if len(source_ids) != len(set(source_ids)):
        raise HumanReviewError("resume gate must not reuse source records across cases")
    public_cases = {item["case_id"]: item for item in packet_cases}
    visible_verdicts: dict[bytes, str] = {}
    for linkage_case in linkage_cases:
        case = public_cases.get(linkage_case["case_id"])
        if case is None:
            raise HumanReviewError("resume gate linkage contains an unknown public case")
        visible_key = _canonical_bytes(
            {"facts": case["facts"], "candidate_statement": case["candidate_statement"]}
        )
        expected_verdict = linkage_case["expected_verdict"]
        previous = visible_verdicts.setdefault(visible_key, expected_verdict)
        if previous != expected_verdict:
            raise HumanReviewError(
                "identical visible case content cannot carry conflicting expected verdicts"
            )
    if len(packet_cases) != RESUME_GATE_PACKET_SIZE:
        raise HumanReviewError("resume gate construction did not produce exactly 75 cases")
    packet_cases.sort(
        key=lambda item: _opaque_id(
            manifest_hash, artifact_hash, seed, "packet-order", item["case_id"]
        )
    )
    packet_id = _opaque_id(
        manifest_hash,
        artifact_hash,
        seed,
        "resume-gate-packet",
        *sorted(item["case_id"] for item in packet_cases),
    )
    packet = {
        "packet_version": PACKET_VERSION,
        "packet_id": packet_id,
        "case_count": RESUME_GATE_PACKET_SIZE,
        "cases": packet_cases,
    }
    linkage = {
        "linkage_version": PACKET_VERSION,
        "packet_id": packet_id,
        "dataset_manifest_sha256": manifest_hash,
        "model_artifact_sha256": artifact_hash,
        "case_count": RESUME_GATE_PACKET_SIZE,
        "design": "proposed_resume_gate_75_v1",
        "cases": sorted(linkage_cases, key=lambda item: item["case_id"]),
    }
    _validate_packet(packet)
    return ReviewExport(packet=packet, private_linkage=linkage)


def main(argv: Sequence[str] | None = None) -> int:
    """Export only the proposed local resume gate; no server or provider is invoked."""
    parser = argparse.ArgumentParser(
        description="Export a proposed, blinded local resume-review packet."
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--private-linkage", type=Path, required=True)
    parser.add_argument("--seed", default=RESUME_GATE_SEED)
    args = parser.parse_args(argv)
    try:
        export = select_resume_gate_packet(
            args.dataset_manifest, args.model_artifact, seed=args.seed
        )
        export_review_artifacts(export, args.packet, args.private_linkage)
    except HumanReviewError as exc:
        print(f"resume gate export failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"case_count": export.packet["case_count"], "packet_id": export.packet["packet_id"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
