import pytest
from pydantic import ValidationError

from reason_code.explanations import EvidenceItem, EvidenceKind, ExplanationEvidence
from reason_code.reason_mapping import (
    MAPPING_CATALOG_VERSION,
    MAPPING_CONTRACT_VERSION,
    DemoMappingEvidenceEnvelope,
    MappingEvidence,
    MappingStatus,
    ReasonMappingEntry,
    resolve_demo_mapping,
)


@pytest.mark.parametrize(
    ("factor", "status", "code"),
    [
        (
            "debt_to_income_ratio",
            MappingStatus.MAPPED,
            "demo_debt_to_income_risk_contribution",
        ),
        ("unmapped_demo_factor", MappingStatus.UNMAPPED, None),
        ("ambiguous_demo_factor", MappingStatus.AMBIGUOUS, None),
        ("ineligible_demo_factor", MappingStatus.INELIGIBLE, None),
        ("unknown_demo_factor", MappingStatus.UNMAPPED, None),
    ],
)
def test_catalog_resolves_each_status_without_inferring_demo_codes(
    factor: str, status: MappingStatus, code: str | None
) -> None:
    mapping = resolve_demo_mapping(factor)

    assert mapping.status is status
    assert mapping.demo_code == code
    assert mapping.catalog_version == MAPPING_CATALOG_VERSION
    assert (mapping.label is not None) is (status is MappingStatus.MAPPED)


def test_catalog_entries_require_complete_mapped_data_and_no_nonmapped_code() -> None:
    with pytest.raises(ValidationError, match="mapped entries require a demo_code and label"):
        ReasonMappingEntry(
            factor="missing_code",
            status=MappingStatus.MAPPED,
            label="Incomplete demo mapping",
            rationale="Test contract validation.",
        )
    with pytest.raises(ValidationError, match="non-mapped entries must not expose"):
        ReasonMappingEntry(
            factor="wrong_code",
            status=MappingStatus.AMBIGUOUS,
            demo_code="demo_wrong_code",
            rationale="Test contract validation.",
        )


def test_mapping_evidence_envelope_is_versioned_closed_and_case_bound() -> None:
    evidence = ExplanationEvidence(
        case_id="demo-case",
        items=(
            EvidenceItem(
                evidence_id="decision",
                kind=EvidenceKind.DECISION,
                label="Demo decision",
                value="decline",
            ),
        ),
    )
    envelope = DemoMappingEvidenceEnvelope(
        case_id="demo-case",
        mappings=(resolve_demo_mapping("debt_to_income_ratio"),),
        explanation_evidence=evidence,
    )

    assert envelope.contract_version == MAPPING_CONTRACT_VERSION
    assert envelope.catalog_version == MAPPING_CATALOG_VERSION
    assert envelope.explanation_evidence == evidence

    with pytest.raises(ValidationError, match="case_id must match explanation evidence"):
        DemoMappingEvidenceEnvelope(
            case_id="other-case",
            mappings=(),
            explanation_evidence=evidence,
        )
    with pytest.raises(ValidationError, match="mapping envelope factors must be unique"):
        DemoMappingEvidenceEnvelope(
            case_id="demo-case",
            mappings=(
                resolve_demo_mapping("debt_to_income_ratio"),
                resolve_demo_mapping("debt_to_income_ratio"),
            ),
            explanation_evidence=evidence,
        )
    with pytest.raises(ValidationError, match="Input should be 'demo-factor-catalog-v1'"):
        DemoMappingEvidenceEnvelope(
            case_id="demo-case",
            mappings=(
                MappingEvidence(
                    factor="debt_to_income_ratio",
                    status=MappingStatus.MAPPED,
                    catalog_version="wrong-catalog-version",
                    demo_code="demo_debt_to_income_risk_contribution",
                    label="Debt-to-income ratio risk contribution",
                ),
            ),
            explanation_evidence=evidence,
        )
