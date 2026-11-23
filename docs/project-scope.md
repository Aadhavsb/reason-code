# Frozen MVP scope

Reason Code is an **auditable decision-support demo**, not lending advice and not production underwriting. It uses public HMDA mortgage data because that dataset contains human-recorded denial reasons; it does not claim that mortgage patterns transfer directly to BNPL, corporate cards, or any other lending product.

## Product contract

The eventual `POST /decisions` route receives a validated, application-shaped record and returns:

1. a calibrated risk estimate and a decision or `human_review` abstention;
2. model-supported factor attributions;
3. mapped adverse-action reason codes;
4. an explanation generated only from that allowed evidence; and
5. programmatic and rubric-based faithfulness verdicts.

## Non-negotiable evidence rules

- Every published metric names the split, evaluation-set version, and assumptions.
- Demographic diagnostics are descriptive slices only, never causal fair-lending claims.
- The explanation layer cannot cite a factor, reason, or number absent from the allowed evidence.
- A frozen, hand-reviewed evaluation set and a CI regression gate are core scope, not polish.

## Explicitly out of scope

Plaid Link, money movement, production bank integration, real underwriting advice, unqualified fairness claims, and placeholder portfolio metrics.
