"""Validate and fingerprint local dataset provenance manifests.

This module records metadata supplied by a researcher. It never downloads data,
checks a remote source, or implies a dataset has been acquired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


class ManifestValidationError(ValueError):
    """Raised when a provenance manifest is incomplete or malformed."""


def _require_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _validate_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ManifestValidationError("source_url must be an absolute HTTP(S) URL")


def _validate_date(value: str) -> None:
    try:
        date.fromisoformat(value)
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


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a local HMDA provenance manifest and return normalized metadata.

    Validation intentionally covers only the manifest's structure. A passing
    result says the metadata is complete and well-formed, not that its source
    was downloaded, verified, representative, or suitable for lending use.
    """
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest must be a JSON object")

    missing = REQUIRED_FIELDS.difference(manifest)
    if missing:
        raise ManifestValidationError(
            f"manifest is missing required fields: {', '.join(sorted(missing))}"
        )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}, "
            f"got {manifest['schema_version']!r}"
        )

    source_url = _require_string(manifest, "source_url")
    _validate_url(source_url)
    download_date = _require_string(manifest, "download_date")
    _validate_date(download_date)

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
        "filter": dict(filters),
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
    """Load and validate a JSON manifest from a local path without fetching URLs."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"manifest file does not exist: {path}") from exc
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
