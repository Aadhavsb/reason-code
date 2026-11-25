#!/usr/bin/env python3
"""Prepare the bounded, local HMDA experiment slice without network access.

The caller supplies each already-downloaded official FFIEC/CFPB CSV explicitly. This
script does not fetch data, train a model, or write into the repository. It emits a
prepared CSV plus the provenance and dataset manifests required by
``reason_code.data.load_local_dataset``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
OUTPUT_CSV_NAME = "hmda-oh-2022-2024-prepared.csv"
PROVENANCE_NAME = "provenance.json"
DATASET_MANIFEST_NAME = "dataset-manifest.json"
ACQUISITION_RECORD_NAME = "acquisition-record.json"
NUMERIC = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z", re.ASCII)
OUTPUT_FIELDS = (
    "loan_id",
    "activity_date",
    "action_taken",
    "income",
    "loan_amount",
    "debt_to_income_ratio",
)
REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "activity_year",
        "lei",
        "state_code",
        "action_taken",
        "loan_type",
        "loan_purpose",
        "lien_status",
        "occupancy_type",
        "business_or_commercial_purpose",
        "income",
        "loan_amount",
        "debt_to_income_ratio",
    }
)


class PreparationError(ValueError):
    """Raised when a raw source or requested output is unsafe to prepare."""


@dataclass(frozen=True)
class RawSource:
    """One downloaded annual official CSV and its documented activity year."""

    year: int
    path: Path


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest through bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(value: str) -> RawSource:
    """Parse one ``YEAR:PATH`` command-line source declaration."""
    try:
        raw_year, raw_path = value.split(":", maxsplit=1)
        year = int(raw_year)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sources must use YEAR:PATH") from exc
    if year < 2018 or year > 2100 or not raw_path:
        raise argparse.ArgumentTypeError("source year or path is invalid")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise argparse.ArgumentTypeError(
            f"source does not name a readable file: {raw_path}"
        ) from exc
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"source does not name a file: {raw_path}")
    return RawSource(year=year, path=path)


def parse_number(value: str) -> float | None:
    """Accept only finite non-negative decimal source values."""
    stripped = value.strip()
    if NUMERIC.fullmatch(stripped) is None:
        return None
    number = float(stripped)
    return number if math.isfinite(number) else None


def is_eligible(row: dict[str, str]) -> bool:
    """Apply the documented, source-level slice filters."""
    return (
        row["state_code"] == "OH"
        and row["action_taken"] in {"1", "3"}
        and row["loan_type"] == "1"
        and row["loan_purpose"] == "1"
        and row["lien_status"] == "1"
        and row["occupancy_type"] == "1"
        and row["business_or_commercial_purpose"] == "2"
    )


def prepared_row(
    row: dict[str, str], *, year: int, source_row_number: int
) -> dict[str, str] | None:
    """Return one normalized non-identifying record, or exclude invalid source data."""
    income_thousands = parse_number(row["income"])
    loan_amount = parse_number(row["loan_amount"])
    dti = parse_number(row["debt_to_income_ratio"])
    if (
        income_thousands is None
        or income_thousands <= 0
        or loan_amount is None
        or loan_amount <= 0
        or dti is None
        or not 0 <= dti <= 100
    ):
        return None
    return {
        "loan_id": f"hmda-{year}-{source_row_number}",
        # HMDA public LAR has activity_year, not a filing-level application date.
        # This is an explicit year-end proxy, not an observed event date.
        "activity_date": f"{year}-12-31",
        "action_taken": row["action_taken"],
        # The HMDA field is reported in thousands of dollars; normalize to dollars.
        "income": format(income_thousands * 1000, ".15g"),
        "loan_amount": format(loan_amount, ".15g"),
        "debt_to_income_ratio": format(dti, ".15g"),
    }


def prepare(sources: tuple[RawSource, ...], output_dir: Path) -> dict[str, Any]:
    """Create a reproducible bounded slice and manifests in an empty output directory."""
    if len({source.year for source in sources}) != len(sources):
        raise PreparationError("each source year must occur exactly once")
    expected_years = (2022, 2023, 2024)
    if tuple(sorted(source.year for source in sources)) != expected_years:
        raise PreparationError(
            "this approved slice requires exactly the 2022, 2023, and 2024 sources"
        )
    if output_dir.exists():
        raise PreparationError("output directory must not already exist")
    output_dir.mkdir(parents=True)

    output_csv = output_dir / OUTPUT_CSV_NAME
    exclusion_counts: Counter[str] = Counter()
    source_records: list[dict[str, Any]] = []
    total_rows = 0
    try:
        with output_csv.open("x", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            for source in sorted(sources, key=lambda item: item.year):
                source_hash = sha256_file(source.path)
                accepted = 0
                source_total = 0
                with source.path.open(encoding="utf-8", newline="") as input_handle:
                    reader = csv.DictReader(input_handle)
                    if reader.fieldnames is None:
                        raise PreparationError(f"{source.path} has no CSV header")
                    missing = REQUIRED_SOURCE_FIELDS.difference(reader.fieldnames)
                    if missing:
                        names = ", ".join(sorted(missing))
                        raise PreparationError(
                            f"{source.path} is missing required columns: {names}"
                        )
                    for source_row_number, row in enumerate(reader, start=2):
                        source_total += 1
                        if row["activity_year"] != str(source.year):
                            raise PreparationError(
                                f"{source.path} row {source_row_number} has an "
                                "unexpected activity_year"
                            )
                        if not is_eligible(row):
                            exclusion_counts["source_filter"] += 1
                            continue
                        result = prepared_row(
                            row, year=source.year, source_row_number=source_row_number
                        )
                        if result is None:
                            exclusion_counts["missing_or_non_numeric_required_value"] += 1
                            continue
                        writer.writerow(result)
                        accepted += 1
                        total_rows += 1
                source_records.append(
                    {
                        "activity_year": source.year,
                        "source_url": SOURCE_URL,
                        "source_sha256": source_hash,
                        "raw_rows": source_total,
                        "accepted_rows": accepted,
                    }
                )
        if total_rows > 250_000:
            raise PreparationError("prepared output exceeds the local loader row limit")
        if total_rows == 0:
            raise PreparationError("slice filters produced no records")

        dataset_hash = sha256_file(output_csv)
        provenance = {
            "schema_version": "1.1",
            "sources": [
                {
                    "activity_year": item["activity_year"],
                    "source_url": item["source_url"],
                    "source_version_hash": item["source_sha256"],
                }
                for item in source_records
            ],
            "filter": {
                "state_code": ["OH"],
                "activity_year": [2022, 2023, 2024],
                "action_taken": ["1", "3"],
                "loan_type": ["1"],
                "loan_purpose": ["1"],
                "lien_status": ["1"],
                "occupancy_type": ["1"],
                "business_or_commercial_purpose": ["2"],
                "income": "numeric, greater than zero; HMDA thousands normalized to dollars",
                "loan_amount": "numeric, greater than zero",
                "debt_to_income_ratio": (
                    "numeric 0..100 only; interval, NA, and exempt values excluded"
                ),
            },
            "download_date": datetime.now(UTC).date().isoformat(),
            "fields": list(OUTPUT_FIELDS),
        }
        dataset_manifest = {
            "schema_version": "1.0",
            "dataset_path": OUTPUT_CSV_NAME,
            "dataset_sha256": dataset_hash,
            "provenance_manifest_path": PROVENANCE_NAME,
            "record_id_column": "loan_id",
            "record_date_column": "activity_date",
            "target_column": "action_taken",
            "target_mapping": {"1": 1, "3": 0},
            "feature_columns": {
                "annual_income": "income",
                "requested_amount": "loan_amount",
                "debt_to_income_ratio": "debt_to_income_ratio",
            },
            "split": {"train_end": "2022-12-31", "validation_end": "2023-12-31"},
        }
        acquisition = {
            "prepared_at_utc": datetime.now(UTC).isoformat(),
            "prepared_csv": OUTPUT_CSV_NAME,
            "prepared_csv_sha256": dataset_hash,
            "prepared_rows": total_rows,
            "sources": source_records,
            "excluded_rows_by_reason": dict(sorted(exclusion_counts.items())),
            "limitations": [
                (
                    "activity_date is a year-end proxy derived from activity_year, "
                    "not an observed event date"
                ),
                "the slice excludes free-form and interval debt-to-income values",
                "the output is a research slice and not suitable for lending decisions",
            ],
        }
        for filename, document in (
            (PROVENANCE_NAME, provenance),
            (DATASET_MANIFEST_NAME, dataset_manifest),
            (ACQUISITION_RECORD_NAME, acquisition),
        ):
            (output_dir / filename).write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return acquisition
    except Exception:
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the approved local Ohio HMDA slice.")
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        help="one explicit YEAR:PATH official raw CSV (repeat for 2022, 2023, and 2024)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="new empty output directory")
    arguments = parser.parse_args()
    try:
        result = prepare(tuple(arguments.source), arguments.output_dir.expanduser())
    except PreparationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
