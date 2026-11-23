# Reason Code

An auditable adverse-action decisioning demo built on public HMDA mortgage data.

> **Decision-support demo, not lending advice or production underwriting.**

Reason Code accepts an application-shaped record, produces a calibrated decision or abstention, maps model-supported factors to an adverse-action reason-code taxonomy, and evaluates whether an applicant-facing explanation stays faithful to that evidence.

## Why it exists

For model-driven credit decisions, a plausible explanation is not enough. The system is designed to make its evidence trace inspectable and to catch unsupported claims before they are presented as an explanation.

## Planned system

```text
HMDA public data → reproducible feature pipeline → calibrated decision model
→ attribution + reason-code mapping → constrained explanation
→ programmatic checks + faithfulness judge → frozen evaluation set + CI gate
```

## Evidence policy

This repository will publish only measurements that identify their dataset split, evaluation-set version, and assumptions. Until a measurement is reproducible, it is intentionally absent.

## Status

Repository scaffold in progress. See [`docs/project-scope.md`](docs/project-scope.md) for the frozen MVP boundary.
