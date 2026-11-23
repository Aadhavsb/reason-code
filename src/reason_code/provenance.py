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
from typing import Any
from urllib.parse import unquote, urlparse

MANIFEST_SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "source_url",
        "filter",
        "download_date",
        "fields",
        "source_version_hash",
    }
)
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_MAX_URL_DECODE_PASSES = 4


class ManifestValidationError(ValueError):
    """Raised when a provenance manifest is incomplete or malformed."""


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
        raise ManifestValidationError(
            "filter must contain only JSON-serializable values"
        ) from exc
    return normalized


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a local HMDA provenance manifest and return normalized metadata.

    Unknown top-level fields are forbidden. Filters are recursively normalized to
    JSON-compatible mappings and sequences; timezone-aware datetimes are rejected
    rather than normalized, so equivalent instants cannot receive different hashes.
    A passing result says only that supplied metadata is complete and well-formed,
    not that its source was downloaded, verified, representative, or suitable for
    lending use.
    """
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest must be a JSON object")

    missing = REQUIRED_FIELDS.difference(manifest)
    if missing:
        raise ManifestValidationError(
            f"manifest is missing required fields: {', '.join(sorted(missing))}"
        )
    unknown = set(manifest).difference(REQUIRED_FIELDS)
    if unknown:
        unknown_names = ", ".join(sorted(unknown))
        raise ManifestValidationError(
            f"manifest contains unsupported fields (unknown fields are forbidden): {unknown_names}"
        )
    schema_version = manifest["schema_version"]
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    source_url = _require_string(manifest, "source_url")
    _validate_url(source_url)
    download_date = _validate_date(_require_string(manifest, "download_date"))

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

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_url": source_url,
        "filter": _validate_filters(filters),
        "download_date": download_date,
        "fields": normalized_fields,
        "source_version_hash": _validate_hash(_require_string(manifest, "source_version_hash")),
    }


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
    try:
        raw = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"manifest is not valid JSON: {exc.msg}") from exc
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
