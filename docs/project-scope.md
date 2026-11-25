# Frozen MVP scope

Reason Code is an **auditable decision-support demo**, not lending advice and not production underwriting. It uses public HMDA mortgage data because that dataset contains human-recorded denial reasons; it does not claim that mortgage patterns transfer directly to BNPL, corporate cards, or any other lending product.

## Product contract

The current API separates the decision and explanation boundaries:

1. `POST /decisions` receives a validated, application-shaped record and returns a deterministic demo risk score, decision or `human_review` abstention, and typed attributions.
2. `POST /decisions/explanation` independently recomputes that bundled demo decision, then returns demo-only reason-code mappings, a closed evidence envelope, a deterministic explanation, and one deterministic faithfulness result.

The explanation endpoint is not a calibrated-model or adverse-action notice contract. It refuses with `422 Unprocessable Content` when its closed demo mapping cannot safely represent the deterministic decision output (for example, an unmapped or duplicate adverse factor).

A future real-data system would require a calibrated risk estimate, validated adverse-action reason-code mapping, and additional evaluation evidence before any production or regulatory claim.

## Non-negotiable evidence rules

- Every published metric names the split, evaluation-set version, and assumptions.
- Demographic diagnostics are descriptive slices only, never causal fair-lending claims.
- The explanation layer cannot cite a factor, reason, or number absent from the allowed evidence.
- A frozen, hand-reviewed evaluation set and a CI regression gate are core scope, not polish.

## Explicitly out of scope

Plaid Link, money movement, production bank integration, real underwriting advice, unqualified fairness claims, and placeholder portfolio metrics.
