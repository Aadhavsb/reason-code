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

## Cost policy

The default development and CI path uses only local deterministic fixtures: no API key, model call, cloud service, automatic dataset download, or paid deployment. Any external data or LLM evaluation is an explicit, manual opt-in. See [`docs/no-cost-policy.md`](docs/no-cost-policy.md).

## Local walkthrough UI

The FastAPI app serves a self-contained, keyboard-accessible case-file UI at `/`.
It is deliberately separate from the CLI and evaluation utilities. Start it locally:

```bash
uvicorn reason_code.main:app --app-dir src --reload
```

Then open <http://127.0.0.1:8000>. The UI submits only to the local `POST /decisions`
route and shows the application-shaped inputs, decision or human-review result, risk
score, score source, policy version, and returned attributions.

### UI boundary

The walkthrough does **not** load or represent an HMDA dataset, call an LLM, or make
external requests. Its bundled scorer is a deterministic demo heuristic, not a
calibrated model. The current decision endpoint also does not return real reason-code
mappings, a generated explanation, a closed explanation-evidence set, or a
faithfulness verdict; the UI calls these omissions out rather than inventing them.

## Status

Repository scaffold in progress. See [`docs/project-scope.md`](docs/project-scope.md) for the frozen MVP boundary.
