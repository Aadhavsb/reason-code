"""Chronological split rules for offline evaluation.

The helper defines boundaries but does not load, shuffle, or infer dates from
HMDA data. Callers must record the resulting split specification with metrics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


class SplitSpecificationError(ValueError):
    """Raised when split boundaries or record dates are invalid."""


_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")


def _is_missing_date(value: object) -> bool:
    """Recognize missing scalar date values, including pandas' NaT, without importing pandas."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _parse_date(value: str | date, label: str) -> date:
    """Return a calendar date from a canonical ISO string or date-like scalar."""
    if _is_missing_date(value):
        raise SplitSpecificationError(f"{label} must not be missing")
    if isinstance(value, datetime):
        normalized = value.date()
        if _is_missing_date(normalized):
            raise SplitSpecificationError(f"{label} must not be missing")
        return normalized
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise SplitSpecificationError(f"{label} must use ISO-8601 YYYY-MM-DD format") from exc
    raise SplitSpecificationError(f"{label} must use ISO-8601 YYYY-MM-DD format or datetime.date")


@dataclass(frozen=True)
class ChronologicalSplitSpec:
    """A deterministic, inclusive-end-date train/validation/test split.

    Records dated on ``train_end`` stay in train; records dated after it through
    ``validation_end`` stay in validation; later records stay in test. Constructor
    inputs are normalized and validated too, so direct construction cannot bypass
    the date and chronological-order invariants.
    """

    train_end: date
    validation_end: date

    def __post_init__(self) -> None:
        train_end = _parse_date(self.train_end, "train_end")
        validation_end = _parse_date(self.validation_end, "validation_end")
        if train_end >= validation_end:
            raise SplitSpecificationError("train_end must be earlier than validation_end")
        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "validation_end", validation_end)

    @classmethod
    def from_iso_dates(cls, train_end: str, validation_end: str) -> ChronologicalSplitSpec:
        return cls(train_end=train_end, validation_end=validation_end)

    def assign(self, record_date: str | date) -> str:
        """Return the split name for one observed-record date."""
        observed = _parse_date(record_date, "record date")
        if observed <= self.train_end:
            return "train"
        if observed <= self.validation_end:
            return "validation"
        return "test"

    def split_records(
        self, records: Iterable[Mapping[str, Any]], *, date_field: str
    ) -> dict[str, list[Mapping[str, Any]]]:
        """Partition records deterministically by a caller-selected date field."""
        result: dict[str, list[Mapping[str, Any]]] = {"train": [], "validation": [], "test": []}
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise SplitSpecificationError(f"record {index} must be a mapping")
            if date_field not in record:
                raise SplitSpecificationError(
                    f"record {index} is missing date field {date_field!r}"
                )
            try:
                split = self.assign(record[date_field])
            except SplitSpecificationError as exc:
                raise SplitSpecificationError(
                    f"record {index} has invalid date field {date_field!r}: {exc}"
                ) from exc
            result[split].append(record)
        return result

    def as_dict(self) -> dict[str, str]:
        """Serialize the contract alongside experiment metadata."""
        return {
            "rule": (
                "train <= train_end; train_end < validation <= validation_end; "
                "test > validation_end"
            ),
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
        }
