"""Validate and fingerprint local dataset provenance manifests.

This module records metadata supplied by a researcher. It never downloads data,
checks a remote source, or implies a dataset has been acquired. Manifests use a
forbid-unknown-fields posture so an unreviewed field cannot silently alter the
meaning of recorded provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse

MANIFEST_SCHEMA_VERSION = "1.1"
LEGACY_MANIFEST_SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "source_url",
        "filter",
        "download_date",
        "fields",
        "source_version_hash",
    }
)
REQUIRED_FIELDS_V1_1 = frozenset(
    {
        "schema_version",
        "sources",
        "filter",
        "download_date",
        "fields",
    }
)
SOURCE_REQUIRED_FIELDS = frozenset({"activity_year", "source_url", "source_version_hash"})
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_MAX_URL_DECODE_PASSES = 4


class ManifestValidationError(ValueError):
    """Raised when a provenance manifest is incomplete or malformed."""


JsonValidationError = TypeVar("JsonValidationError", bound=ValueError)


def parse_json_object(
    contents: str, *, error_type: type[JsonValidationError], subject: str
) -> Mapping[str, Any]:
    """Parse one JSON object while rejecting duplicate keys at every nesting level.

    Python's default decoder silently keeps the last duplicate key. Manifests are
    review inputs, so that ambiguity must fail before schema validation. ``error_type``
    preserves each caller's public validation-error contract.
    """

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise error_type(f"{subject} contains duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        raw = json.loads(contents, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise error_type(f"{subject} is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise error_type(f"{subject} must be a JSON object")
    return raw


def _require_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{key} must be a non-empty string")
    return value


def _has_whitespace_or_control(value: str) -> bool:
    return any(
        character.isspace() or unicodedata.category(character) == "Cc" for character in value
    )


def _decode_url_until_stable(value: str) -> str:
    """Decode URL escapes to a bounded fixed point for safety checks."""
    decoded = value
    for _ in range(_MAX_URL_DECODE_PASSES):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            return decoded
        decoded = next_decoded
    raise ManifestValidationError("source_url contains excessive nested percent-encoding")


def _validate_url(value: str) -> None:
    decoded = _decode_url_until_stable(value)
    if _has_whitespace_or_control(value) or _has_whitespace_or_control(decoded):
        raise ManifestValidationError(
            "source_url must not contain whitespace or control characters"
        )

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ManifestValidationError("source_url must be an absolute HTTP(S) URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ManifestValidationError("source_url must be an absolute HTTP(S) URL") from exc
    if port == 0:
        raise ManifestValidationError("source_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ManifestValidationError("source_url must not include embedded credentials")
    if parsed.params:
        raise ManifestValidationError("source_url must not include path parameters")
    if parsed.query or parsed.fragment:
        raise ManifestValidationError(
            "source_url must not include query parameters or fragments "
            "because they can embed tokens"
        )


def _validate_date(value: str) -> str:
    if not _ISO_DATE.fullmatch(value):
        raise ManifestValidationError("download_date must use ISO-8601 YYYY-MM-DD format")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ManifestValidationError("download_date must use ISO-8601 YYYY-MM-DD format") from exc


def _validate_hash(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ManifestValidationError(
            "source_version_hash must be a 64-character SHA-256 hex digest"
        )
    return normalized


def _normalize_filter_value(value: Any) -> Any:
    """Copy mappings and sequences while enforcing JSON-object key constraints."""
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ManifestValidationError("filter mapping keys must be non-empty strings")
            normalized[key] = _normalize_filter_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_filter_value(item) for item in value]
    return value


def _validate_filters(filters: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe filter object with canonical mapping and sequence types.

    Filters deliberately accept JSON values only: date and datetime objects, including
    timezone-aware datetimes, are rejected rather than normalized. Callers must supply
    an explicit JSON string, which prevents equivalent instants from acquiring distinct
    manifest hashes through timezone formatting.
    """
    try:
        normalized = _normalize_filter_value(filters)
        json.dumps(normalized, allow_nan=False)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ManifestValidationError("filter must contain only JSON-serializable values") from exc
    return normalized


def _validate_sources(value: Any) -> list[dict[str, Any]]:
    """Validate an ordered, non-ambiguous list of official source artifacts."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ManifestValidationError("sources must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen_years: set[int] = set()
    for source in value:
        if not isinstance(source, Mapping):
            raise ManifestValidationError("each source must be a JSON object")
        missing = SOURCE_REQUIRED_FIELDS.difference(source)
        unknown = set(source).difference(SOURCE_REQUIRED_FIELDS)
        if missing or unknown:
            raise ManifestValidationError(
                "each source must contain exactly activity_year, source_url, "
                "and source_version_hash"
            )
        year = source["activity_year"]
        if type(year) is not int or not 2017 <= year <= 2100 or year in seen_years:
            raise ManifestValidationError(
                "source activity_year values must be unique integers from 2017 through 2100"
            )
        source_url = _require_string(source, "source_url")
        _validate_url(source_url)
        normalized.append(
            {
                "activity_year": year,
                "source_url": source_url,
                "source_version_hash": _validate_hash(
                    _require_string(source, "source_version_hash")
                ),
            }
        )
        seen_years.add(year)
    return normalized


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a local HMDA provenance manifest and return normalized metadata.

    Schema ``1.0`` records one source and remains readable for existing local
    manifests. Schema ``1.1`` records several independently fingerprinted official
    source artifacts, which avoids inventing one hash for a multi-year slice.
    """
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    if schema_version == LEGACY_MANIFEST_SCHEMA_VERSION:
        required_fields = REQUIRED_FIELDS_V1
    elif schema_version == MANIFEST_SCHEMA_VERSION:
        required_fields = REQUIRED_FIELDS_V1_1
    else:
        raise ManifestValidationError(
            "schema_version must be "
            f"{LEGACY_MANIFEST_SCHEMA_VERSION!r} or {MANIFEST_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    missing = required_fields.difference(manifest)
    if missing:
        raise ManifestValidationError(
            f"manifest is missing required fields: {', '.join(sorted(missing))}"
        )
    unknown = set(manifest).difference(required_fields)
    if unknown:
        unknown_names = ", ".join(sorted(unknown))
        raise ManifestValidationError(
            f"manifest contains unsupported fields (unknown fields are forbidden): {unknown_names}"
        )

    filters = manifest["filter"]
    if not isinstance(filters, Mapping):
        raise ManifestValidationError(
            "filter must be a JSON object; use {} when no filters are applied"
        )
    fields = manifest["fields"]
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence) or not fields:
        raise ManifestValidationError("fields must be a non-empty list of field names")
    normalized_fields: list[str] = []
    for field in fields:
        if not isinstance(field, str) or not field.strip():
            raise ManifestValidationError("fields must contain only non-empty strings")
        normalized_fields.append(field.strip())
    if len(set(normalized_fields)) != len(normalized_fields):
        raise ManifestValidationError("fields must not contain duplicates")

    normalized = {
        "schema_version": schema_version,
        "filter": _validate_filters(filters),
        "download_date": _validate_date(_require_string(manifest, "download_date")),
        "fields": normalized_fields,
    }
    if schema_version == LEGACY_MANIFEST_SCHEMA_VERSION:
        source_url = _require_string(manifest, "source_url")
        _validate_url(source_url)
        normalized["source_url"] = source_url
        normalized["source_version_hash"] = _validate_hash(
            _require_string(manifest, "source_version_hash")
        )
    else:
        normalized["sources"] = _validate_sources(manifest["sources"])
    return normalized


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint of a validated manifest's contents."""
    normalized = validate_manifest(manifest)
    canonical_json = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a local JSON manifest from a local path without fetching URLs."""
    try:
        contents = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ManifestValidationError(f"manifest file cannot be read: {path}") from exc
    raw = parse_json_object(contents, error_type=ManifestValidationError, subject="manifest")
    return validate_manifest(raw)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local manifest validator or canonical manifest hasher."""
    parser = argparse.ArgumentParser(
        description="Validate or hash a local HMDA provenance manifest."
    )
    parser.add_argument("command", choices=("validate", "hash"))
    parser.add_argument("manifest", type=Path, help="path to a local JSON provenance manifest")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestValidationError as exc:
        parser.error(str(exc))

    if args.command == "validate":
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(manifest_hash(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
