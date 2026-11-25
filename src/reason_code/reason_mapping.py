"""Versioned, demo-only mapping from supported factors to demo reason codes.

The catalog is a research contract for the bundled deterministic demo.  It is not an
adverse-action, ECOA, HMDA, or other regulatory taxonomy, and it must not be used to
translate outputs from a real-data research model into ``POST /decisions``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reason_code.explanations import ExplanationEvidence

MAPPING_CONTRACT_VERSION = "demo-evidence-reason-mapping-v1"
MAPPING_CATALOG_VERSION = "demo-factor-catalog-v1"


class MappingStatus(StrEnum):
    """Disposition of a factor under this deliberately narrow demo catalog."""

    MAPPED = "mapped"
    UNMAPPED = "unmapped"
    AMBIGUOUS = "ambiguous"
    INELIGIBLE = "ineligible"


class ReasonMappingEntry(BaseModel):
    """One catalog entry; only ``mapped`` entries may expose a demo reason code."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    factor: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    status: MappingStatus
    demo_code: str | None = Field(default=None, pattern=r"^demo_[a-z0-9_]+$")
    label: str | None = Field(default=None, min_length=1, max_length=128)
    rationale: str = Field(min_length=1, max_length=280)

    @model_validator(mode="after")
    def code_and_label_match_status(self) -> ReasonMappingEntry:
        has_mapping = self.demo_code is not None or self.label is not None
        if self.status is MappingStatus.MAPPED and (
            self.demo_code is None or self.label is None
        ):
            raise ValueError("mapped entries require a demo_code and label")
        if self.status is not MappingStatus.MAPPED and has_mapping:
            raise ValueError("non-mapped entries must not expose a demo reason code")
        return self


class MappingEvidence(BaseModel):
    """Closed mapping evidence for one factor in a single demo explanation."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    factor: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    status: MappingStatus
    catalog_version: Literal["demo-factor-catalog-v1"] = MAPPING_CATALOG_VERSION
    demo_code: str | None = Field(default=None, pattern=r"^demo_[a-z0-9_]+$")
    label: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def mapped_evidence_is_complete(self) -> MappingEvidence:
        has_mapping = self.demo_code is not None or self.label is not None
        if self.status is MappingStatus.MAPPED and (
            self.demo_code is None or self.label is None
        ):
            raise ValueError("mapped evidence requires a demo_code and label")
        if self.status is not MappingStatus.MAPPED and has_mapping:
            raise ValueError("non-mapped evidence must not expose a demo reason code")
        return self


class DemoMappingEvidenceEnvelope(BaseModel):
    """Versioned, closed evidence envelope for a demo mapping response only."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    contract_version: Literal["demo-evidence-reason-mapping-v1"] = MAPPING_CONTRACT_VERSION
    catalog_version: Literal["demo-factor-catalog-v1"] = MAPPING_CATALOG_VERSION
    mappings: tuple[MappingEvidence, ...]
    explanation_evidence: ExplanationEvidence

    @model_validator(mode="after")
    def envelope_is_case_bound_and_unique(self) -> DemoMappingEvidenceEnvelope:
        if self.case_id != self.explanation_evidence.case_id:
            raise ValueError("mapping envelope case_id must match explanation evidence")
        factors = [mapping.factor for mapping in self.mappings]
        if len(factors) != len(set(factors)):
            raise ValueError("mapping envelope factors must be unique")
        if any(mapping.catalog_version != self.catalog_version for mapping in self.mappings):
            raise ValueError("mapping evidence must use the envelope catalog version")
        return self


_CATALOG_ENTRIES = (
    ReasonMappingEntry(
        factor="debt_to_income_ratio",
        status=MappingStatus.MAPPED,
        demo_code="demo_debt_to_income_risk_contribution",
        label="Debt-to-income ratio risk contribution",
        rationale="Supported deterministic demo factor.",
    ),
    ReasonMappingEntry(
        factor="requested_amount_to_income",
        status=MappingStatus.MAPPED,
        demo_code="demo_requested_amount_to_income_risk_contribution",
        label="Requested amount relative to income risk contribution",
        rationale="Supported deterministic demo factor.",
    ),
    ReasonMappingEntry(
        factor="credit_score_deficiency",
        status=MappingStatus.MAPPED,
        demo_code="demo_credit_score_deficiency_risk_contribution",
        label="Credit score deficiency risk contribution",
        rationale="Supported deterministic demo factor.",
    ),
    ReasonMappingEntry(
        factor="unmapped_demo_factor",
        status=MappingStatus.UNMAPPED,
        rationale="The catalog has no demo reason-code representation for this factor.",
    ),
    ReasonMappingEntry(
        factor="ambiguous_demo_factor",
        status=MappingStatus.AMBIGUOUS,
        rationale="The factor has more than one plausible demo interpretation.",
    ),
    ReasonMappingEntry(
        factor="ineligible_demo_factor",
        status=MappingStatus.INELIGIBLE,
        rationale="The factor is explicitly outside the demo explanation boundary.",
    ),
)
DEMO_REASON_MAPPING_CATALOG = {entry.factor: entry for entry in _CATALOG_ENTRIES}


def resolve_demo_mapping(factor: str) -> MappingEvidence:
    """Resolve one factor without inferring a mapping for unknown factors.

    Unknown factors deliberately resolve to ``unmapped`` rather than receiving a
    best-effort code.  Callers must fail closed before producing a reason-code output.
    """
    entry = DEMO_REASON_MAPPING_CATALOG.get(factor)
    if entry is None:
        return MappingEvidence(factor=factor, status=MappingStatus.UNMAPPED)
    return MappingEvidence(
        factor=entry.factor,
        status=entry.status,
        demo_code=entry.demo_code,
        label=entry.label,
    )
