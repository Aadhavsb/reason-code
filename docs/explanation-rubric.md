# Explanation faithfulness rubric

This module checks whether an explanation is bounded by the explicit evidence supplied for one case. It is a deterministic regression guard for an auditable decision-support demo; it is not a model-quality score, legal determination, fairness assessment, or lending recommendation.

## Boundary

The generator accepts `ExplanationEvidence`, a closed set of typed `EvidenceItem` records. It renders one claim per input item in input order and keeps a claim trace containing the cited evidence ID, kind, subject, and rendered statement. It does not rank factors, infer causality, add context, or generate values that are absent from those records.

The programmatic grader evaluates the trace, not free-form provenance inference. For every claim it requires:

1. a cited evidence ID that exists in the case;
2. the claim kind to exactly match that item's kind;
3. the claim subject to exactly match that item's label after case and whitespace normalization; and
4. every numeric literal in the claim statement to appear in the cited evidence item's label, value, or detail, with percentage status retained.

The verdict is machine-readable: `faithful` is `false` whenever `reasons` is nonempty. Reason codes are `unknown_evidence_id`, `evidence_kind_mismatch`, `unsupported_subject`, and `unsupported_number`.

## What this does not claim

This checker cannot establish that upstream model attributions are correct, that a reason code is legally sufficient, or that arbitrary unstructured prose is semantically faithful. Upstream evidence production and any human/legal review remain separate responsibilities.

`tests/fixtures/explanation_eval_synthetic_scaffold.json` is a tiny frozen, hand-reviewed-style **synthetic scaffold**. It exists solely to make behavior reproducible and must not be represented as reported gold-set performance or evaluation results.
