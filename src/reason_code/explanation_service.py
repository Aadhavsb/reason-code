"""Fail-closed construction of deterministic demo explanation responses."""

from reason_code.explanations import (
    EvidenceItem,
    EvidenceKind,
    ExplanationEvidence,
    generate_explanation,
    grade_faithfulness,
)
from reason_code.reason_mapping import (
    DemoMappingEvidenceEnvelope,
    MappingEvidence,
    MappingStatus,
    resolve_demo_mapping,
)
from reason_code.schemas import (
    DEMO_EXPLANATION_NOTICE,
    DEMO_SCORE_SOURCE,
    AttributionDirection,
    DecisionExplanationResponse,
    DecisionResponse,
    DemoReasonCode,
)


class ExplanationUnavailableError(ValueError):
    """Raised when a decision cannot safely use the closed demo explanation path."""


def build_decision_explanation(decision: DecisionResponse) -> DecisionExplanationResponse:
    """Build a verified explanation for the bundled heuristic or fail without output."""
    if decision.score_source != DEMO_SCORE_SOURCE:
        raise ExplanationUnavailableError(
            "Deterministic explanations are available only for the bundled demo heuristic."
        )

    adverse_attributions = [
        attribution
        for attribution in decision.attributions
        if attribution.direction is AttributionDirection.INCREASES_RISK
        and attribution.contribution > 0
    ]
    adverse_factors = [attribution.factor for attribution in adverse_attributions]
    if len(adverse_factors) != len(set(adverse_factors)):
        raise ExplanationUnavailableError(
            "Duplicate adverse attribution factors cannot be represented in a "
            "deterministic explanation."
        )

    reason_codes: list[DemoReasonCode] = []
    mapping_evidence: list[MappingEvidence] = []
    evidence_items = [
        EvidenceItem(
            evidence_id="decision",
            kind=EvidenceKind.DECISION,
            label="Demo decision",
            value=decision.decision.value,
        )
    ]
    for attribution in adverse_attributions:
        mapping = resolve_demo_mapping(attribution.factor)
        if mapping.status is not MappingStatus.MAPPED:
            raise ExplanationUnavailableError(
                "No deterministic demo reason-code mapping exists for an adverse attribution."
            )
        if mapping.demo_code is None or mapping.label is None:  # Defensive catalog boundary.
            raise ExplanationUnavailableError("Demo mapping catalog entry is incomplete.")
        mapping_evidence.append(mapping)
        reason_code = DemoReasonCode(
            code=mapping.demo_code, factor=attribution.factor, label=mapping.label
        )
        reason_codes.append(reason_code)
        evidence_items.extend(
            (
                EvidenceItem(
                    evidence_id=f"factor_{attribution.factor}",
                    kind=EvidenceKind.FACTOR,
                    label=mapping.label,
                    value=attribution.contribution,
                ),
                EvidenceItem(
                    evidence_id=f"reason_{attribution.factor}",
                    kind=EvidenceKind.REASON_CODE,
                    label="Demo reason code",
                    value=reason_code.code,
                ),
            )
        )

    evidence = ExplanationEvidence(case_id=decision.application_id, items=tuple(evidence_items))
    mapping_envelope = DemoMappingEvidenceEnvelope(
        case_id=decision.application_id,
        mappings=tuple(mapping_evidence),
        explanation_evidence=evidence,
    )
    explanation = generate_explanation(evidence)
    faithfulness = grade_faithfulness(explanation, evidence)
    if not faithfulness.faithful:
        raise ExplanationUnavailableError(
            "Deterministic explanation failed its faithfulness checks."
        )
    return DecisionExplanationResponse(
        decision=decision,
        reason_codes=tuple(reason_codes),
        evidence=evidence,
        mapping_envelope=mapping_envelope,
        explanation=explanation,
        faithfulness=faithfulness,
        explanation_notice=DEMO_EXPLANATION_NOTICE,
    )
