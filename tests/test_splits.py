from datetime import date, datetime

import pandas as pd
import pytest

from reason_code.splits import ChronologicalSplitSpec, SplitSpecificationError


def test_assigns_dates_using_documented_inclusive_boundaries():
    spec = ChronologicalSplitSpec.from_iso_dates("2023-12-31", "2024-12-31")

    assert spec.assign("2023-12-31") == "train"
    assert spec.assign(date(2024, 1, 1)) == "validation"
    assert spec.assign("2024-12-31") == "validation"
    assert spec.assign("2025-01-01") == "test"


def test_normalizes_datetime_and_pandas_timestamp_to_calendar_dates():
    spec = ChronologicalSplitSpec(datetime(2023, 12, 31, 23, 59), pd.Timestamp("2024-12-31T12:00"))

    assert spec.train_end == date(2023, 12, 31)
    assert spec.validation_end == date(2024, 12, 31)
    assert spec.assign(datetime(2024, 1, 1, 1, 30)) == "validation"
    assert spec.assign(pd.Timestamp("2025-01-01T00:00:00Z")) == "test"


@pytest.mark.parametrize("value", [None, pd.NaT])
def test_rejects_missing_dates(value):
    spec = ChronologicalSplitSpec.from_iso_dates("2023-12-31", "2024-12-31")

    with pytest.raises(SplitSpecificationError, match="must not be missing"):
        spec.assign(value)


def test_constructor_enforces_date_and_chronological_invariants():
    with pytest.raises(SplitSpecificationError, match="ISO-8601"):
        ChronologicalSplitSpec("20231231", "2024-12-31")
    with pytest.raises(SplitSpecificationError, match="earlier"):
        ChronologicalSplitSpec(date(2024, 12, 31), date(2024, 12, 31))


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


def test_split_records_reports_record_index_for_missing_and_invalid_dates():
    spec = ChronologicalSplitSpec.from_iso_dates("2023-12-31", "2024-12-31")

    with pytest.raises(
        SplitSpecificationError, match=r"record 0 is missing date field 'activity_date'"
    ):
        spec.split_records([{"id": "unknown"}], date_field="activity_date")
    with pytest.raises(
        SplitSpecificationError, match=r"record 1 has invalid date field 'activity_date'"
    ):
        spec.split_records(
            [{"activity_date": "2023-01-01"}, {"activity_date": None}], date_field="activity_date"
        )


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
