import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from reason_code.provenance import (
    ManifestValidationError,
    load_manifest,
    manifest_hash,
    validate_manifest,
)


@pytest.fixture
def manifest():
    return {
        "schema_version": "1.0",
        "source_url": "https://example.gov/hmda/2025.csv",
        "filter": {"activity_year": 2025, "state_code": ["OH"]},
        "download_date": "2026-11-23",
        "fields": ["activity_year", "action_taken"],
        "source_version_hash": "a" * 64,
    }


def test_validate_manifest_normalizes_hash_and_preserves_required_metadata(manifest):
    manifest["source_version_hash"] = "A" * 64

    validated = validate_manifest(manifest)

    assert validated["source_version_hash"] == "a" * 64
    assert validated["filter"] == {"activity_year": 2025, "state_code": ["OH"]}


def test_manifest_hash_is_stable_across_json_key_order(manifest):
    reordered = dict(reversed(list(manifest.items())))

    assert manifest_hash(manifest) == manifest_hash(reordered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_url", "ftp://example.gov/hmda.csv", r"absolute HTTP\(S\) URL"),
        ("download_date", "23-11-2026", "ISO-8601"),
        ("fields", [], "non-empty list"),
        ("source_version_hash", "not-a-hash", "64-character SHA-256"),
    ],
)
def test_manifest_rejects_invalid_required_values(manifest, field, value, message):
    manifest[field] = value

    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest(manifest)


@pytest.mark.parametrize("value", ["20261123", "2026-1-23", "2026-11-23T00:00:00", " 2026-11-23"])
def test_manifest_requires_canonical_download_date(manifest, value):
    manifest["download_date"] = value

    with pytest.raises(ManifestValidationError, match="ISO-8601"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://user:password@example.gov/hmda.csv", "embedded credentials"),
        ("https://example.gov/hmda.csv?access_token=secret", "embed tokens"),
        ("https://example.gov/hmda.csv#token", "embed tokens"),
        ("https://example.gov/a file.csv", "whitespace or control"),
        ("https://example.gov/hmda%0Acsv", "whitespace or control"),
        ("https://example.gov/hmda%2520csv", "whitespace or control"),
        ("https://example.gov/hmda%250Acsv", "whitespace or control"),
        ("https://example.gov/hmda.csv;version=1", "path parameters"),
    ],
)
def test_manifest_rejects_unsafe_source_urls(manifest, url, message):
    manifest["source_url"] = url

    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest(manifest)


def test_manifest_forbids_unknown_fields(manifest):
    manifest["unreviewed_override"] = True

    with pytest.raises(ManifestValidationError, match="unknown fields are forbidden"):
        validate_manifest(manifest)


def test_manifest_requires_filter_object(manifest):
    manifest["filter"] = "activity_year=2025"

    with pytest.raises(ManifestValidationError, match="filter must be a JSON object"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "filter_value",
    [
        {"nested": {"": "value"}},
        {"nested": {1: "value"}},
        {"unsupported": object()},
        {"not_a_number": float("nan")},
    ],
)
def test_manifest_rejects_non_json_or_invalid_nested_filter_values(manifest, filter_value):
    manifest["filter"] = filter_value

    with pytest.raises(ManifestValidationError, match="filter"):
        validate_manifest(manifest)


def test_manifest_normalizes_nested_filter_mappings_and_sequences(manifest):
    manifest["filter"] = {"nested": {"years": (2024, 2025)}}

    validated = validate_manifest(manifest)

    assert validated["filter"] == {"nested": {"years": [2024, 2025]}}


@pytest.mark.parametrize(
    "aware_datetime",
    [
        datetime(2026, 11, 23, 12, tzinfo=UTC),
        datetime(2026, 11, 23, 7, tzinfo=timezone(timedelta(hours=-5))),
    ],
)
def test_manifest_rejects_equivalent_aware_datetimes_in_filters(manifest, aware_datetime):
    manifest["filter"] = {"observed_at": aware_datetime}

    with pytest.raises(ManifestValidationError, match="JSON-serializable"):
        validate_manifest(manifest)


@pytest.mark.parametrize("filename", ["missing.json", "directory"])
def test_load_manifest_translates_read_failures(tmp_path: Path, filename):
    if filename == "directory":
        (tmp_path / filename).mkdir()

    with pytest.raises(ManifestValidationError, match="cannot be read"):
        load_manifest(tmp_path / filename)


def test_load_manifest_translates_invalid_utf8(tmp_path: Path):
    manifest_path = tmp_path / "invalid-utf8.json"
    manifest_path.write_bytes(b"\xff")

    with pytest.raises(ManifestValidationError, match="cannot be read"):
        load_manifest(manifest_path)


def test_module_cli_validates_and_hashes_without_network_access(tmp_path: Path, manifest):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    project_root = Path(__file__).parents[1]
    environment = {"PYTHONPATH": str(project_root / "src")}

    validation = subprocess.run(
        [sys.executable, "-m", "reason_code.provenance", "validate", str(manifest_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    digest = subprocess.run(
        [sys.executable, "-m", "reason_code.provenance", "hash", str(manifest_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(validation.stdout)["source_url"] == manifest["source_url"]
    assert digest.stdout.strip() == manifest_hash(manifest)
