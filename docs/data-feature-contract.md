# Local dataset and feature contract

> This is a contract for a future **local** experiment. The repository includes only a three-row synthetic fixture and does not claim to have acquired HMDA data, trained a model, or measured a result.

## Deliberate boundary

`reason_code.data.load_local_dataset()` requires an explicit path to a local dataset manifest. It has no default manifest, data discovery, download behavior, provider call, or training behavior. The supplied manifest must name a local CSV and local provenance manifest, and it must include the CSV's SHA-256 digest. The loader validates the linked provenance manifest with `reason_code.provenance`, then streams and verifies the CSV bytes in bounded memory before parsing through that same open file handle.

The returned data is isolated from the demo `POST /decisions` scorer. Loading it is preparation and validation only; it neither fits a model nor changes decision behavior.

## Dataset manifest schema `1.0`

All keys are required and unknown keys fail validation:

```json
{
  "schema_version": "1.0",
  "dataset_path": "prepared-slice.csv",
  "dataset_sha256": "<64 lowercase-or-uppercase SHA-256 hex characters>",
  "provenance_manifest_path": "provenance.json",
  "record_id_column": "loan_id",
  "record_date_column": "activity_date",
  "target_column": "action_taken",
  "target_mapping": {"1": 1, "3": 0},
  "feature_columns": {
    "annual_income": "income",
    "requested_amount": "loan_amount",
    "debt_to_income_ratio": "debt_to_income_ratio"
  },
  "split": {"train_end": "2023-12-31", "validation_end": "2024-12-31"}
}
```

Relative file paths resolve from the manifest's directory; absolute paths are also local paths. The linked provenance manifest must list every selected raw column in its `fields` array. It may use provenance schema `1.0` for one source or `1.1` for independently fingerprinted annual source artifacts. This preserves a reviewable connection between source metadata and the small prepared slice without asserting that the metadata was independently verified.

## Model-ready feature contract

Only these three normalized feature names may enter a later training workflow:

| Normalized feature | Required value | Validation |
| --- | --- | --- |
| `annual_income` | Numeric annual-income value using a documented source unit/transform | finite and `> 0` |
| `requested_amount` | Numeric requested loan amount using a documented source unit/transform | finite and `> 0` |
| `debt_to_income_ratio` | Numeric percentage, not a free-form HMDA interval string | finite and `0..100` inclusive |

The selected raw columns must be distinct from each other, `record_id_column`, and `target_column`. Every selected value is required. Record IDs must be unique and may not contain Unicode control characters. The target mapping must map observed source strings to both binary labels `0` and `1`; values outside that mapping fail closed. Any unselected CSV columns are ignored and cannot silently become model features.

CSV input is decoded as UTF-8; one leading UTF-8 BOM on the header is accepted. Feature values must use a decimal lexical form (optional sign, decimal point, and exponent); Python-specific forms such as `1_000`, hexadecimal floats, `nan`, and `inf` are rejected before conversion. Parsed values must still satisfy the feature range checks below.

The loader fail-closes at **250,000 data rows** (`MAX_DATASET_ROWS`), a deliberate ceiling for a local prepared HMDA slice rather than a full public filing. A larger dataset must be explicitly prepared into a reviewed slice; the loader will not silently truncate or partially load it.

`record_date_column` must contain canonical `YYYY-MM-DD` dates. `activity_year` alone is not a valid record date: a future preparation step must document how it produced a date precise enough for this contract. Missing or malformed dates fail before partitioning.

## Chronological partition contract

The loader calls the existing `ChronologicalSplitSpec.split_records()` utility using normalized `record_date` values. Its exact rule is:

```text
train <= train_end
train_end < validation <= validation_end
test > validation_end
```

A load fails unless every train, validation, and test partition contains at least one row. The loader returns records plus partitions; it does not shuffle, fit, calibrate, evaluate, or report metrics. Any future experiment must persist the manifest hash, source provenance hash, split specification, transformations, model configuration, and measured metrics separately before making a claim.

## Local synthetic smoke test

The committed fixture is intentionally tiny and synthetic. It exists solely to test parser, hash, validation, and split wiring:

```bash
PYTHONPATH=src python3 -c 'from reason_code.data import load_local_dataset; print(load_local_dataset("tests/fixtures/local_dataset_manifest.json").partitions.keys())'
```

Do not replace it with downloaded HMDA data or use it to claim model performance.
