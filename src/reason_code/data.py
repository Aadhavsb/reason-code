"""Local-only, fail-closed loading for a future versioned dataset slice.

This module deliberately does not download, discover, or train on data. Callers must
supply a local manifest path, and the manifest must name and fingerprint a local CSV.
It returns validated records and chronological partitions only; it does not feed the
demo decision endpoint.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

from reason_code.provenance import ManifestValidationError, load_manifest, parse_json_object
from reason_code.splits import ChronologicalSplitSpec, SplitSpecificationError

DATASET_MANIFEST_SCHEMA_VERSION = "1.0"
FEATURE_NAMES = frozenset({"annual_income", "requested_amount", "debt_to_income_ratio"})
_HASH_CHUNK_SIZE = 1024 * 1024
# A local HMDA experiment slice must remain bounded: reject rather than partially load.
MAX_DATASET_ROWS = 250_000
_NUMBER_LEXEME = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z",
    re.ASCII,
)
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_path",
        "dataset_sha256",
        "provenance_manifest_path",
        "record_id_column",
        "record_date_column",
        "target_column",
        "target_mapping",
        "feature_columns",
        "split",
    }
)


class DatasetValidationError(ValueError):
    """Raised when a local dataset manifest or its selected data is invalid."""


def _freeze(value: Any) -> Any:
    """Return an immutable, defensive copy of JSON-like validation output."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class LocalDatasetManifest:
    """Validated local dataset metadata required to prepare a future experiment."""

    dataset_path: Path
    dataset_sha256: str
    provenance_manifest_path: Path
    record_id_column: str
    record_date_column: str
    target_column: str
    target_mapping: Mapping[str, int]
    feature_columns: Mapping[str, str]
    split: ChronologicalSplitSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_mapping", _freeze(self.target_mapping))
        object.__setattr__(self, "feature_columns", _freeze(self.feature_columns))


@dataclass(frozen=True)
class DatasetSplits:
    """Immutable validated model-ready records and chronological partitions."""

    manifest: LocalDatasetManifest
    provenance: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    partitions: Mapping[str, tuple[Mapping[str, Any], ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        object.__setattr__(self, "records", tuple(_freeze(record) for record in self.records))
        object.__setattr__(self, "partitions", _freeze(self.partitions))


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    normalized = _non_empty_string(value, field).lower()
    is_hex_digest = len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )
    if not is_hex_digest:
        raise DatasetValidationError(f"{field} must be a 64-character SHA-256 hex digest")
    return normalized


def _local_file(value: Any, manifest_path: Path, field: str) -> Path:
    try:
        candidate = Path(_non_empty_string(value, field)).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        resolved = candidate.resolve(strict=True)
        is_file = resolved.is_file()
    except (OSError, RuntimeError, ValueError) as exc:
        raise DatasetValidationError(f"{field} does not name a readable local file") from exc
    if not is_file:
        raise DatasetValidationError(f"{field} does not name a readable local file")
    return resolved


def _column_name(value: Any, field: str) -> str:
    name = _non_empty_string(value, field)
    if "\n" in name or "\r" in name:
        raise DatasetValidationError(f"{field} must be a single CSV column name")
    return name


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DatasetValidationError(f"dataset manifest cannot be read as JSON: {path}") from exc
    try:
        return parse_json_object(
            contents, error_type=DatasetValidationError, subject="dataset manifest"
        )
    except DatasetValidationError as exc:
        raise DatasetValidationError(
            f"dataset manifest cannot be read as JSON: {path}: {exc}"
        ) from exc


def load_dataset_manifest(path: str | Path) -> LocalDatasetManifest:
    """Load one explicitly supplied local data manifest without any network access."""
    try:
        manifest_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DatasetValidationError(f"dataset manifest cannot be read: {path}") from exc
    raw = _load_json_object(manifest_path)
    missing = _REQUIRED_MANIFEST_FIELDS.difference(raw)
    unknown = set(raw).difference(_REQUIRED_MANIFEST_FIELDS)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise DatasetValidationError(f"dataset manifest is missing fields: {missing_names}")
    if unknown:
        unknown_names = ", ".join(sorted(unknown))
        raise DatasetValidationError(
            f"dataset manifest contains unsupported fields: {unknown_names}"
        )
    if raw["schema_version"] != DATASET_MANIFEST_SCHEMA_VERSION:
        version = raw["schema_version"]
        raise DatasetValidationError(
            f"schema_version must be {DATASET_MANIFEST_SCHEMA_VERSION!r}, got {version!r}"
        )

    target_mapping = raw["target_mapping"]
    if not isinstance(target_mapping, Mapping) or not target_mapping:
        raise DatasetValidationError("target_mapping must be a non-empty JSON object")
    normalized_target_mapping: dict[str, int] = {}
    for source_value, label in target_mapping.items():
        key = _non_empty_string(source_value, "target_mapping keys")
        if type(label) is not int or label not in {0, 1}:
            raise DatasetValidationError(
                "target_mapping values must be integer binary labels (0 or 1)"
            )
        if key in normalized_target_mapping:
            raise DatasetValidationError(
                "target_mapping keys must be unique after whitespace normalization"
            )
        normalized_target_mapping[key] = label
    if set(normalized_target_mapping.values()) != {0, 1}:
        raise DatasetValidationError("target_mapping must include both binary labels 0 and 1")

    feature_columns = raw["feature_columns"]
    if not isinstance(feature_columns, Mapping) or set(feature_columns) != FEATURE_NAMES:
        names = ", ".join(sorted(FEATURE_NAMES))
        raise DatasetValidationError(f"feature_columns must contain exactly: {names}")
    normalized_features = {
        name: _column_name(column, f"feature_columns.{name}")
        for name, column in feature_columns.items()
    }
    selected_columns = [
        _column_name(raw["record_id_column"], "record_id_column"),
        _column_name(raw["record_date_column"], "record_date_column"),
        _column_name(raw["target_column"], "target_column"),
        *normalized_features.values(),
    ]
    if len(set(selected_columns)) != len(selected_columns):
        raise DatasetValidationError("record, target, and feature columns must be distinct")

    split = raw["split"]
    if not isinstance(split, Mapping) or set(split) != {"train_end", "validation_end"}:
        raise DatasetValidationError("split must contain exactly train_end and validation_end")
    try:
        split_spec = ChronologicalSplitSpec.from_iso_dates(
            split["train_end"], split["validation_end"]
        )
    except (KeyError, TypeError, SplitSpecificationError) as exc:
        raise DatasetValidationError(f"invalid chronological split: {exc}") from exc

    return LocalDatasetManifest(
        dataset_path=_local_file(raw["dataset_path"], manifest_path, "dataset_path"),
        dataset_sha256=_sha256(raw["dataset_sha256"], "dataset_sha256"),
        provenance_manifest_path=_local_file(
            raw["provenance_manifest_path"], manifest_path, "provenance_manifest_path"
        ),
        record_id_column=selected_columns[0],
        record_date_column=selected_columns[1],
        target_column=selected_columns[2],
        target_mapping=normalized_target_mapping,
        feature_columns=normalized_features,
        split=split_spec,
    )


def _verify_dataset_hash(handle: BinaryIO, expected: str) -> None:
    """Verify a dataset through one bounded-memory pass over an already-open handle."""
    hasher = hashlib.sha256()
    while chunk := handle.read(_HASH_CHUNK_SIZE):
        hasher.update(chunk)
    if hasher.hexdigest() != expected:
        raise DatasetValidationError("dataset_sha256 does not match the local dataset file")
    handle.seek(0)


def _required_value(row: Mapping[str, str | None], column: str, row_number: int) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise DatasetValidationError(f"row {row_number} has a missing value for {column!r}")
    return value.strip()


def _finite_number(value: str, feature: str, row_number: int) -> float:
    if _NUMBER_LEXEME.fullmatch(value) is None:
        raise DatasetValidationError(
            f"row {row_number} has a non-numeric value for feature {feature!r}"
        )
    try:
        number = float(value)
    except ValueError as exc:  # Defensive boundary if Python's parser changes.
        raise DatasetValidationError(
            f"row {row_number} has a non-numeric value for feature {feature!r}"
        ) from exc
    if not math.isfinite(number):
        raise DatasetValidationError(
            f"row {row_number} has a non-finite value for feature {feature!r}"
        )
    return number


def _validate_record_id(record_id: str, row_number: int) -> None:
    if any(unicodedata.category(character) == "Cc" for character in record_id):
        raise DatasetValidationError(
            f"row {row_number} record id must not contain control characters"
        )


def _validate_feature_ranges(record: Mapping[str, Any], row_number: int) -> None:
    if record["annual_income"] <= 0:
        raise DatasetValidationError(f"row {row_number} annual_income must be greater than zero")
    if record["requested_amount"] <= 0:
        raise DatasetValidationError(f"row {row_number} requested_amount must be greater than zero")
    if not 0 <= record["debt_to_income_ratio"] <= 100:
        raise DatasetValidationError(
            f"row {row_number} debt_to_income_ratio must be between 0 and 100"
        )


def _load_records(manifest: LocalDatasetManifest) -> tuple[dict[str, Any], ...]:
    required_columns = {
        manifest.record_id_column,
        manifest.record_date_column,
        manifest.target_column,
        *manifest.feature_columns.values(),
    }
    try:
        # Hash and parse through the same descriptor so path replacement cannot create a
        # hash-then-reopen gap. ``utf-8-sig`` accepts exactly a leading UTF-8 BOM.
        with manifest.dataset_path.open("rb") as binary_handle:
            _verify_dataset_hash(binary_handle, manifest.dataset_sha256)
            with io.TextIOWrapper(binary_handle, encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise DatasetValidationError("dataset CSV must include a header row")
                headers = [
                    header.strip() if header is not None else "" for header in reader.fieldnames
                ]
                if (
                    not all(headers)
                    or len(set(headers)) != len(headers)
                    or headers != reader.fieldnames
                ):
                    raise DatasetValidationError(
                        "dataset CSV headers must be non-empty, unique, and free of "
                        "surrounding whitespace"
                    )
                missing = required_columns.difference(headers)
                if missing:
                    missing_names = ", ".join(sorted(missing))
                    raise DatasetValidationError(
                        f"dataset CSV is missing required columns: {missing_names}"
                    )

                records: list[dict[str, Any]] = []
                identifiers: set[str] = set()
                for row_number, row in enumerate(reader, start=2):
                    if row_number - 1 > MAX_DATASET_ROWS:
                        raise DatasetValidationError(
                            f"dataset CSV exceeds fail-closed row limit of {MAX_DATASET_ROWS}"
                        )
                    if None in row:
                        raise DatasetValidationError(
                            f"row {row_number} has more values than its header"
                        )
                    record_id = _required_value(row, manifest.record_id_column, row_number)
                    _validate_record_id(record_id, row_number)
                    if record_id in identifiers:
                        raise DatasetValidationError(
                            f"row {row_number} duplicates record id {record_id!r}"
                        )
                    identifiers.add(record_id)
                    raw_target = _required_value(row, manifest.target_column, row_number)
                    if raw_target not in manifest.target_mapping:
                        raise DatasetValidationError(
                            f"row {row_number} has target value {raw_target!r} outside "
                            "target_mapping"
                        )
                    record: dict[str, Any] = {
                        "record_id": record_id,
                        "record_date": _required_value(
                            row, manifest.record_date_column, row_number
                        ),
                        "label": manifest.target_mapping[raw_target],
                    }
                    for feature, column in manifest.feature_columns.items():
                        raw_value = _required_value(row, column, row_number)
                        record[feature] = _finite_number(raw_value, feature, row_number)
                    _validate_feature_ranges(record, row_number)
                    try:
                        manifest.split.assign(record["record_date"])
                    except SplitSpecificationError as exc:
                        raise DatasetValidationError(
                            f"row {row_number} has invalid record_date: {exc}"
                        ) from exc
                    records.append(record)
    except DatasetValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError("dataset file cannot be read as UTF-8 CSV") from exc
    if not records:
        raise DatasetValidationError("dataset CSV must contain at least one data row")
    return tuple(records)


def load_local_dataset(path: str | Path) -> DatasetSplits:
    """Load, validate, and chronologically partition a manifest-specified local CSV.

    ``path`` is intentionally required: there is no default manifest, bundled data
    source, remote URL fetch, or implicit filesystem discovery.
    """
    manifest = load_dataset_manifest(path)
    try:
        provenance = load_manifest(manifest.provenance_manifest_path)
    except ManifestValidationError as exc:
        raise DatasetValidationError(f"invalid linked provenance manifest: {exc}") from exc
    selected_columns = {
        manifest.record_id_column,
        manifest.record_date_column,
        manifest.target_column,
        *manifest.feature_columns.values(),
    }
    if not selected_columns.issubset(set(provenance["fields"])):
        raise DatasetValidationError(
            "linked provenance manifest fields must include every selected column"
        )
    records = _load_records(manifest)
    try:
        split_records = manifest.split.split_records(records, date_field="record_date")
    except SplitSpecificationError as exc:  # Defensive boundary around the shared utility.
        raise DatasetValidationError(f"could not partition validated records: {exc}") from exc
    partitions = {name: tuple(items) for name, items in split_records.items()}
    if any(not partitions[name] for name in ("train", "validation", "test")):
        raise DatasetValidationError(
            "dataset must contain at least one record in each chronological split"
        )
    return DatasetSplits(
        manifest=manifest,
        provenance=provenance,
        records=records,
        partitions=partitions,
    )
