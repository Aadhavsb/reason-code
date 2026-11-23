from datetime import date

import pytest

from reason_code.splits import ChronologicalSplitSpec, SplitSpecificationError


def test_assigns_dates_using_documented_inclusive_boundaries():
    spec = ChronologicalSplitSpec.from_iso_dates("2023-12-31", "2024-12-31")

    assert spec.assign("2023-12-31") == "train"
    assert spec.assign(date(2024, 1, 1)) == "validation"
    assert spec.assign("2024-12-31") == "validation"
    assert spec.assign("2025-01-01") == "test"


def test_split_records_preserves_records_and_uses_selected_date_field():
    spec = ChronologicalSplitSpec.from_iso_dates("2023-12-31", "2024-12-31")
    records = [
        {"id": "train", "activity_date": "2023-10-01"},
        {"id": "validation", "activity_date": "2024-05-01"},
        {"id": "test", "activity_date": "2025-02-01"},
    ]

    split = spec.split_records(records, date_field="activity_date")

    assert [record["id"] for record in split["train"]] == ["train"]
    assert [record["id"] for record in split["validation"]] == ["validation"]
    assert [record["id"] for record in split["test"]] == ["test"]


def test_rejects_invalid_split_order_and_missing_record_dates():
    with pytest.raises(SplitSpecificationError, match="earlier"):
        ChronologicalSplitSpec.from_iso_dates("2024-12-31", "2024-12-31")

    spec = ChronologicalSplitSpec.from_iso_dates("2023-12-31", "2024-12-31")
    with pytest.raises(SplitSpecificationError, match="missing date field"):
        spec.split_records([{"id": "unknown"}], date_field="activity_date")


def test_serializes_split_contract_for_experiment_evidence():
    spec = ChronologicalSplitSpec.from_iso_dates("2023-12-31", "2024-12-31")

    assert spec.as_dict() == {
        "rule": (
            "train <= train_end; train_end < validation <= validation_end; "
            "test > validation_end"
        ),
        "train_end": "2023-12-31",
        "validation_end": "2024-12-31",
    }
