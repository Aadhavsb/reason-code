# Reason Code

> **Evidence-bound explanations for model-assisted credit decisions.**
> A local, auditable decision-support demo—not lending advice, production underwriting, or a compliance determination.

Reason Code is a deliberately constrained credit-decision explanation system. It starts with an application-shaped input, applies a declared decision policy, preserves the factors that supported that result, and tests whether an explanation stays within that evidence.

The point is not to make a model sound convincing. The point is to make it difficult for an explanation to say something the evidence does not support.

```text
application-shaped input
  → demo score / decision / human-review abstention
  → typed attributions + policy version
  → closed evidence envelope + claim trace
  → constrained explanation path
  → deterministic faithfulness checks (blocking)
  → optional LLM-as-judge (advisory)
  → synthetic-fixture regression check in CI
```

## Why this exists

A credit explanation can be fluent, specific, and still wrong. For example, a sentence can cite a real factor but invent the number attached to it. That is a dangerous failure mode in any system where a person may act on the explanation.

Reason Code treats the explanation as an **evidence-bound output**, not free-form prose:

- An explanation may cite only evidence items supplied in a typed, closed envelope.
- Every claim carries a trace to the evidence it used.
- Deterministic checks reject unsupported factors, statements, subjects, and numbers.
- A separate LLM judge can assess clarity or flag issues, but it is never the authority on whether an output is evidence-faithful.

## Local walkthrough UI

The FastAPI app serves a self-contained, keyboard-accessible case-file UI at `/`. It is deliberately separate from the CLI and evaluation utilities:

```bash
uvicorn reason_code.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>. The UI submits only to the local `POST /decisions` route and shows application-shaped inputs, the decision or human-review result, risk score, score source, policy version, and returned attributions.

### UI boundary

The walkthrough does **not** load or represent an HMDA dataset, call an LLM, or make external requests. Its bundled scorer is a deterministic demo heuristic, not a calibrated model. The current decision endpoint also does not return real reason-code mappings, a generated explanation, a closed explanation-evidence set, or a faithfulness verdict; the UI calls these omissions out rather than inventing them.

## Current implementation

The repository intentionally distinguishes working engineering boundaries from data/model claims that require real evidence.

### Built and locally testable

- FastAPI `POST /decisions` endpoint with strict request validation, JSON-safe `422` errors, dependency injection, and an explicit demo notice.
- Deterministic demo scorer labeled `demo_heuristic_not_calibrated`—it does **not** claim a trained or calibrated HMDA model.
- Decision policy with `approve`, `human_review`, and `decline` outcomes. The middle band is an explicit abstention path, not an automated decision.
- Immutable threshold configuration and policy versions bound to the applied threshold values.
- Typed attributions with risk-direction semantics; decline responses require adverse, non-zero supporting evidence.
- Evidence-envelope explanation models, deterministic generator, claim trace, and structural/prose faithfulness graders.
- Optional constrained manual LLM generator and advisory judge adapters, guarded by typed output validation and the deterministic graders.
- Recorded synthetic judge fixture and a deterministic synthetic-fixture evaluation CLI.
- Strict local provenance-manifest validation and deterministic chronological-split utilities.
- GitHub Actions quality workflow: lint, tests, and deterministic evaluation—with no credentials, provider calls, dataset downloads, deployments, or service containers.

### Deliberately not claimed yet

- No HMDA data slice has been acquired or published in this repository.
- No model has been trained, fitted, or calibrated on HMDA data.
- No real adverse-action reason-code taxonomy/mapping has been validated.
- No frozen human-reviewed gold set, judge–human agreement, Cohen’s kappa, fairness result, or public model/faithfulness metric exists.
- No claim of regulatory compliance, lending suitability, loss reduction, or production readiness is made.

The current fixture baseline is explicitly marked **synthetic scaffold — not a real metric**. It proves regression wiring, not product performance.

## Quick start — no keys, no network, no spend

Requirements: Python 3.11+.

```bash
git clone https://github.com/Aadhavsb/reason-code.git
cd reason-code
python3 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m ruff check .
python -m pytest
python -m reason_code.evaluation.run --mode deterministic
```

The default development, test, and CI path is local and fixture-backed. It does not need an account, an API key, an LLM provider, an external database, a cloud deployment, or an HMDA download.

For the full engineering and interview walkthrough, see [`docs/no-cost-policy.md`](docs/no-cost-policy.md) and the vault runbook referenced in the project planning notes.

## Run the local API

Start the FastAPI service:

```bash
uvicorn reason_code.main:app --reload --port 8000
```

Then send a synthetic demonstration request:

```bash
curl -sS http://127.0.0.1:8000/decisions \
  -H 'content-type: application/json' \
  -d '{
    "application_id": "demo-001",
    "annual_income": 100000,
    "requested_amount": 20000,
    "debt_to_income_ratio": 25,
    "credit_score": 720
  }'
```

The response includes:

- `decision`: `approve`, `human_review`, or `decline`;
- `risk_score`: a demo heuristic score, not a calibrated probability;
- `score_source`: always inspect this before describing the result;
- `policy_version`: the threshold configuration that made the decision; and
- `attributions`: typed factor contributions supporting the score.

### Decision policy

The bundled demo policy uses these declared constants:

```text
risk score < 0.40       → approve
0.40 ≤ risk score ≤ 0.60 → human_review
risk score > 0.60       → decline
```

Those values are demonstration policy settings, not optimized lending thresholds. The human-review band exists to show that a decision-support system can abstain rather than force automation at every score.

## Evaluation and regression gate

Run the deterministic evaluation directly:

```bash
python -m reason_code.evaluation.run --mode deterministic
```

The command validates a frozen synthetic fixture against its SHA-256 manifest, regenerates/grades its cases deterministically, checks expected verdicts and reason-code lists, verifies aggregate totals, and emits one machine-readable JSON report.

The CLI is intentionally fail-closed:

- fixture/schema/checksum drift returns a non-zero exit;
- stdout still carries a structured failure verdict;
- diagnostics go to stderr; and
- output is ASCII-safe so locale differences do not turn a valid run into an encoding failure.

GitHub Actions runs the same deterministic command after Ruff and pytest on every push and pull request. It is a quality check; it is not a claim that branch protection or a production release gate is configured.

## Optional LLM path: manual, constrained, and off by default

An LLM is part of the project architecture, but it is not permitted to become an unobserved, paid, or CI-only dependency.

### Default modes

- **Disabled:** no judge/provider is constructed.
- **Recorded:** synthetic, fixture-backed responses exercise the judge seam without a network call.
- **Manual:** the only mode allowed to make a live request, and it requires explicit provider/model selection.

### Allowed manual providers

| Provider | Boundary |
| --- | --- |
| Ollama | Local HTTP loopback only: `localhost`, `127.0.0.1`, or `::1`; no API key. |
| Groq | Explicit manual selection; runtime `GROQ_API_KEY`; fixed provider endpoint only. |
| Gemini | Explicit manual selection; runtime `GEMINI_API_KEY`; fixed provider endpoint only. |

There is **no automatic fallback** between providers. OpenAI, Anthropic, Azure, arbitrary remote endpoints, and provider names outside the approved boundary are rejected. Keys are read only at runtime for the chosen manual provider; they do not belong in Git, CI, test fixtures, screenshots, or documentation.

A live LLM output is still rejected unless it passes:

1. typed `GeneratedExplanation` validation;
2. deterministic prose-to-evidence checks; and
3. deterministic structural faithfulness checks.

The LLM judge is useful for advisory review, but it is not a substitute for the deterministic evidence guardrail and is not yet calibrated against human labels.

## Data governance utilities

Reason Code includes local-only utilities intended to make future real-data work reproducible and reviewable:

```bash
# Once a real local JSON provenance manifest exists:
python -m reason_code.provenance validate /path/to/manifest.json
python -m reason_code.provenance hash /path/to/manifest.json
```

The manifest validator rejects unknown fields, malformed dates, unsafe/token-bearing URLs, non-canonical hashes, and non-JSON-safe filters. It never downloads, verifies, or endorses a remote source.

The chronological split helper encodes:

```text
train <= train_end
train_end < validation <= validation_end
test > validation_end
```

It prevents an obvious form of temporal leakage when real dated records exist. It does not, by itself, prove a modeling choice is valid.

## Repository map

```text
src/reason_code/
├── api.py                 # FastAPI endpoint and JSON-safe validation handling
├── decision.py            # Decision policy and policy-version binding
├── scoring.py             # Injectable scorer protocol + honest demo heuristic
├── schemas.py             # Input, attribution, decision contracts and invariants
├── provenance.py          # Local provenance-manifest validation/hash CLI
├── splits.py              # Chronological split contract
├── explanations/          # Evidence envelope, generator, claim trace, graders
└── evaluation/            # Deterministic runner + recorded/manual judge boundaries

tests/                     # Fixture-backed unit, API, evaluation, and boundary tests
docs/                      # Scope, no-cost policy, and dataset-card template
.github/workflows/ci.yml   # Hermetic lint/test/deterministic-eval workflow
```

## Development principles

1. **Evidence before eloquence.** An explanation that reads well but adds unsupported facts is a failure.
2. **The safe path is the default.** Tests and CI are credential-free, deterministic, and offline.
3. **No implicit cost.** A live provider requires manual selection; there is no fallback.
4. **Abstention is a feature.** A system may route uncertain cases to human review rather than automate every outcome.
5. **No fabricated evidence.** Metrics, calibration, fairness claims, or portfolio copy require a documented dataset split, evaluation-set version, assumptions, and reproducible artifact.
6. **Demo boundaries are product features.** The code names the demo scorer honestly and rejects the kinds of configuration that would quietly turn an interview project into a billable cloud dependency.

## Roadmap and evidence gates

The next meaningful milestones are not cosmetic:

1. acquire a reproducible public HMDA slice with a concrete provenance manifest;
2. build and evaluate a chronologically split, calibrated model;
3. implement and validate a real reason-code mapping layer;
4. create a frozen, human-reviewed evaluation set;
5. measure explanation faithfulness and judge–human agreement with stated assumptions; and
6. add only then a results table, portfolio case study, and resume bullet.

Until then, Reason Code is best described as a well-tested **auditable decision/explanation architecture and evaluation boundary**, not a validated credit model.

## License and use

This repository is an educational decision-support demonstration. Do not use it to make lending decisions, issue adverse-action notices, or infer regulatory compliance.
