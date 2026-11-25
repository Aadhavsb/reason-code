import csv
import hashlib
import json
from pathlib import Path

import pytest

import reason_code.data as data_module
from reason_code.data import DatasetValidationError, load_dataset_manifest, load_local_dataset

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES / "local_dataset_manifest.json"


def test_loads_explicit_local_manifest_validates_csv_and_uses_chronological_splits():
    dataset = load_local_dataset(MANIFEST_PATH)

    assert [row["record_id"] for row in dataset.partitions["train"]] == ["synthetic-train"]
    assert [row["record_id"] for row in dataset.partitions["validation"]] == [
        "synthetic-validation"
    ]
    assert [row["record_id"] for row in dataset.partitions["test"]] == ["synthetic-test"]
    assert dataset.records[0] == {
        "record_id": "synthetic-train",
        "record_date": "2023-06-30",
        "label": 1,
        "annual_income": 85000.0,
        "requested_amount": 180000.0,
        "debt_to_income_ratio": 28.0,
    }


def _write_manifest(tmp_path: Path, **changes: object) -> Path:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["dataset_path"] = str(FIXTURES / "local_dataset.csv")
    manifest["provenance_manifest_path"] = str(FIXTURES / "local_dataset_provenance.json")
    manifest.update(changes)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_rejects_unknown_manifest_keys_and_no_implicit_default_path(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, unreviewed_override=True)

    with pytest.raises(DatasetValidationError, match="unsupported fields"):
        load_dataset_manifest(manifest_path)
    with pytest.raises(TypeError):
        load_local_dataset()  # type: ignore[call-arg]


def test_rejects_dataset_hash_mismatch(tmp_path: Path):
    csv_path = tmp_path / "slice.csv"
    source = (FIXTURES / "local_dataset.csv").read_text(encoding="utf-8")
    csv_path.write_text(source + "\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, dataset_path=str(csv_path))

    with pytest.raises(DatasetValidationError, match="dataset_sha256 does not match"):
        load_local_dataset(manifest_path)


def test_rejects_unmapped_target_and_invalid_feature_data(tmp_path: Path):
    csv_path = tmp_path / "slice.csv"
    csv_path.write_text(
        "loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
        "a,2023-06-30,9,85000,180000,28\n"
        "b,2024-06-30,3,0,210000,34\n"
        "c,2025-06-30,1,110000,250000,22\n",
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path,
        dataset_path=str(csv_path),
        dataset_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(DatasetValidationError, match="outside target_mapping"):
        load_local_dataset(manifest_path)


def test_rejects_invalid_feature_range(tmp_path: Path):
    csv_path = tmp_path / "slice.csv"
    csv_path.write_text(
        "loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
        "a,2023-06-30,1,85000,180000,28\n"
        "b,2024-06-30,3,0,210000,34\n"
        "c,2025-06-30,1,110000,250000,22\n",
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path,
        dataset_path=str(csv_path),
        dataset_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(DatasetValidationError, match="annual_income must be greater than zero"):
        load_local_dataset(manifest_path)


def test_rejects_missing_chronological_partition(tmp_path: Path):
    csv_path = tmp_path / "slice.csv"
    csv_path.write_text(
        "loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
        "a,2023-06-30,1,85000,180000,28\n"
        "b,2023-07-30,3,92000,210000,34\n",
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path,
        dataset_path=str(csv_path),
        dataset_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(
        DatasetValidationError, match="at least one record in each chronological split"
    ):
        load_local_dataset(manifest_path)


def test_rejects_selected_columns_absent_from_linked_provenance(tmp_path: Path):
    provenance_path = tmp_path / "provenance.json"
    fixture_path = FIXTURES / "local_dataset_provenance.json"
    provenance = json.loads(fixture_path.read_text(encoding="utf-8"))
    provenance["fields"].remove("income")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, provenance_manifest_path=str(provenance_path))

    with pytest.raises(DatasetValidationError, match="must include every selected column"):
        load_local_dataset(manifest_path)


def _write_dataset_manifest(tmp_path: Path, contents: bytes) -> Path:
    csv_path = tmp_path / "slice.csv"
    csv_path.write_bytes(contents)
    return _write_manifest(
        tmp_path,
        dataset_path=str(csv_path),
        dataset_sha256=hashlib.sha256(contents).hexdigest(),
    )


def test_accepts_utf8_bom_in_csv_header(tmp_path: Path):
    contents = b"\xef\xbb\xbf" + (FIXTURES / "local_dataset.csv").read_bytes()
    manifest_path = _write_dataset_manifest(tmp_path, contents)

    dataset = load_local_dataset(manifest_path)

    assert dataset.records[0]["record_id"] == "synthetic-train"


@pytest.mark.parametrize("value", ["1_000", "0x1p2", "nan", "inf"])
def test_rejects_python_only_or_non_decimal_numeric_lexemes(tmp_path: Path, value: str):
    contents = (
        (FIXTURES / "local_dataset.csv").read_text(encoding="utf-8").replace("85000", value, 1)
    )
    manifest_path = _write_dataset_manifest(tmp_path, contents.encode("utf-8"))

    with pytest.raises(DatasetValidationError, match="non-numeric value"):
        load_local_dataset(manifest_path)


def test_converts_lazy_utf8_decode_errors_to_dataset_validation_errors(tmp_path: Path):
    header = b"loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
    valid_rows = b"".join(
        f"id-{index},2023-06-30,1,85000,180000,28\n".encode()
        for index in range(500)
    )
    manifest_path = _write_dataset_manifest(tmp_path, header + valid_rows + b"invalid,\xff\n")

    with pytest.raises(DatasetValidationError, match="cannot be read as UTF-8 CSV"):
        load_local_dataset(manifest_path)


def test_converts_lazy_csv_errors_to_dataset_validation_errors(tmp_path: Path, monkeypatch):
    manifest_path = _write_dataset_manifest(
        tmp_path, (FIXTURES / "local_dataset.csv").read_bytes()
    )

    class FailingReader:
        fieldnames = [
            "loan_id",
            "activity_date",
            "action_taken",
            "income",
            "loan_amount",
            "debt_to_income_ratio",
        ]

        def __iter__(self):
            return self

        def __next__(self):
            raise csv.Error("simulated CSV iteration failure")

    monkeypatch.setattr(data_module.csv, "DictReader", lambda _handle: FailingReader())

    with pytest.raises(DatasetValidationError, match="cannot be read as UTF-8 CSV"):
        load_local_dataset(manifest_path)


def test_converts_hash_read_errors_to_dataset_validation_errors(tmp_path: Path, monkeypatch):
    contents = (FIXTURES / "local_dataset.csv").read_bytes()
    manifest_path = _write_dataset_manifest(tmp_path, contents)
    csv_path = tmp_path / "slice.csv"
    original_open = Path.open

    class FailingHashRead:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int) -> bytes:
            raise OSError("simulated read failure")

    def failing_open(self: Path, mode: str = "r", *args, **kwargs):
        if self == csv_path and mode == "rb":
            return FailingHashRead()
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(DatasetValidationError, match="cannot be read as UTF-8 CSV"):
        load_local_dataset(manifest_path)


def test_hashes_and_parses_through_one_dataset_file_handle(tmp_path: Path, monkeypatch):
    contents = (FIXTURES / "local_dataset.csv").read_bytes()
    manifest_path = _write_dataset_manifest(tmp_path, contents)
    csv_path = tmp_path / "slice.csv"
    original_open = Path.open
    opened_modes: list[str] = []

    def counting_open(self: Path, mode: str = "r", *args, **kwargs):
        if self == csv_path:
            opened_modes.append(mode)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)

    assert load_local_dataset(manifest_path).records
    assert opened_modes == ["rb"]


def test_rejects_record_id_control_characters_and_missing_required_headers(tmp_path: Path):
    control_id_contents = (
        b"loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
        b"bad\x1f-id,2023-06-30,1,85000,180000,28\n"
        b"b,2024-06-30,3,92000,210000,34\n"
        b"c,2025-06-30,1,110000,250000,22\n"
    )
    control_id_manifest = _write_dataset_manifest(tmp_path, control_id_contents)

    with pytest.raises(
        DatasetValidationError, match="record id must not contain control characters"
    ):
        load_local_dataset(control_id_manifest)

    missing_header_contents = (
        b"loan_id,activity_date,action_taken,loan_amount,debt_to_income_ratio\n"
        b"a,2023-06-30,1,180000,28\n"
    )
    missing_header_manifest = _write_dataset_manifest(tmp_path, missing_header_contents)

    with pytest.raises(DatasetValidationError, match="missing required columns: income"):
        load_local_dataset(missing_header_manifest)


def test_rejects_target_mapping_key_collisions_after_normalization(tmp_path: Path):
    manifest_path = _write_manifest(
        tmp_path,
        target_mapping={"1": 0, " 1 ": 1, "3": 1},
    )

    with pytest.raises(DatasetValidationError, match="unique after whitespace normalization"):
        load_dataset_manifest(manifest_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "0.9"}, "schema_version must be"),
        ({"target_mapping": {}}, "target_mapping must be a non-empty JSON object"),
        ({"target_mapping": {"1": 0, "3": 0}}, "must include both binary labels"),
        (
            {"target_mapping": {"1": True, "3": 0}},
            "values must be integer binary labels",
        ),
    ],
)
def test_rejects_invalid_schema_and_target_mapping_contracts(
    tmp_path: Path, changes: dict[str, object], message: str
):
    manifest_path = _write_manifest(tmp_path, **changes)

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset_manifest(manifest_path)


@pytest.mark.parametrize("method_name", ["expanduser", "resolve"])
def test_wraps_manifest_path_expansion_and_resolution_errors(
    tmp_path: Path, monkeypatch, method_name: str
):
    manifest_path = _write_manifest(tmp_path)

    def fail_path_operation(*_args, **_kwargs):
        raise RuntimeError("simulated path failure")

    monkeypatch.setattr(Path, method_name, fail_path_operation)

    with pytest.raises(DatasetValidationError, match="dataset manifest cannot be read"):
        load_dataset_manifest(manifest_path)


def test_wraps_referenced_local_path_expansion_errors(tmp_path: Path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)
    original_expanduser = Path.expanduser
    dataset_path = FIXTURES / "local_dataset.csv"

    def fail_dataset_expansion(self: Path):
        if self == dataset_path:
            raise RuntimeError("simulated path failure")
        return original_expanduser(self)

    monkeypatch.setattr(Path, "expanduser", fail_dataset_expansion)

    with pytest.raises(
        DatasetValidationError, match="dataset_path does not name a readable local file"
    ):
        load_dataset_manifest(manifest_path)


def test_wraps_linked_provenance_validation_errors(tmp_path: Path, monkeypatch):
    manifest_path = _write_manifest(tmp_path)

    def fail_provenance(_path: Path):
        raise data_module.ManifestValidationError("simulated malformed provenance")

    monkeypatch.setattr(data_module, "load_manifest", fail_provenance)

    with pytest.raises(DatasetValidationError, match="invalid linked provenance manifest"):
        load_local_dataset(manifest_path)


def test_rejects_duplicate_json_keys_in_dataset_manifest_at_parse_time(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}', encoding="utf-8"
    )

    with pytest.raises(DatasetValidationError, match="duplicate JSON key"):
        load_dataset_manifest(manifest_path)


def test_rejects_nested_duplicate_json_keys_in_dataset_manifest_at_parse_time(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            '"target_mapping": {"1": 1, "3": 0}',
            '"target_mapping": {"1": 1, "1": 0, "3": 0}',
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="duplicate JSON key"):
        load_dataset_manifest(manifest_path)


def test_enforces_fail_closed_dataset_row_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(data_module, "MAX_DATASET_ROWS", 2)
    manifest_path = _write_dataset_manifest(tmp_path, (FIXTURES / "local_dataset.csv").read_bytes())

    with pytest.raises(DatasetValidationError, match="exceeds fail-closed row limit of 2"):
        load_local_dataset(manifest_path)


def test_returns_defensive_read_only_mapping_state():
    dataset = load_local_dataset(MANIFEST_PATH)

    with pytest.raises(TypeError):
        dataset.manifest.target_mapping["9"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        dataset.provenance["fields"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        dataset.records[0]["label"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        dataset.partitions["train"] = ()  # type: ignore[index]
    with pytest.raises(AttributeError):
        dataset.provenance["fields"].append("unexpected")


def test_rejects_malformed_csv_header_and_extra_row_values(tmp_path: Path):
    whitespace_header_contents = (
        b"loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio \n"
        b"a,2023-06-30,1,85000,180000,28\n"
    )
    whitespace_header_manifest = _write_dataset_manifest(tmp_path, whitespace_header_contents)

    with pytest.raises(DatasetValidationError, match="headers must be non-empty, unique"):
        load_local_dataset(whitespace_header_manifest)

    extra_value_contents = (
        b"loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
        b"a,2023-06-30,1,85000,180000,28,unexpected\n"
    )
    extra_value_manifest = _write_dataset_manifest(tmp_path, extra_value_contents)

    with pytest.raises(DatasetValidationError, match="more values than its header"):
        load_local_dataset(extra_value_manifest)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("loan_id,loan_id,action_taken,income,loan_amount,debt_to_income_ratio", "unique"),
        ("loan_id,,action_taken,income,loan_amount,debt_to_income_ratio", "non-empty"),
    ],
)
def test_rejects_duplicate_and_empty_csv_headers(tmp_path: Path, header: str, message: str):
    manifest_path = _write_dataset_manifest(
        tmp_path, f"{header}\na,2023-06-30,1,85000,180000,28\n".encode()
    )

    with pytest.raises(DatasetValidationError, match=message):
        load_local_dataset(manifest_path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("a,,1,85000,180000,28", "missing value for 'activity_date'"),
        ("a,2023-06-30,1,,180000,28", "missing value for 'income'"),
        ("a,not-a-date,1,85000,180000,28", "invalid record_date"),
    ],
)
def test_rejects_missing_or_malformed_required_row_values(tmp_path: Path, row: str, message: str):
    contents = (
        "loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
        f"{row}\n"
    ).encode()
    manifest_path = _write_dataset_manifest(tmp_path, contents)

    with pytest.raises(DatasetValidationError, match=message):
        load_local_dataset(manifest_path)


def test_rejects_duplicate_record_ids(tmp_path: Path):
    contents = (
        b"loan_id,activity_date,action_taken,income,loan_amount,debt_to_income_ratio\n"
        b"a,2023-06-30,1,85000,180000,28\n"
        b"a,2024-06-30,3,92000,210000,34\n"
    )
    manifest_path = _write_dataset_manifest(tmp_path, contents)

    with pytest.raises(DatasetValidationError, match="duplicates record id 'a'"):
        load_local_dataset(manifest_path)
