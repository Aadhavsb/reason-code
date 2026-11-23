import json
import subprocess
import sys
from pathlib import Path

import pytest

from reason_code.provenance import ManifestValidationError, manifest_hash, validate_manifest


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


def test_manifest_requires_filter_object(manifest):
    manifest["filter"] = "activity_year=2025"

    with pytest.raises(ManifestValidationError, match="filter must be a JSON object"):
        validate_manifest(manifest)


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
