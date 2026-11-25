# Demo explanation mapping contract

> **Research-only, local, and non-authoritative.** This versioned contract maps supported deterministic-demo factors to `demo_` identifiers. It is not an adverse-action reason-code taxonomy and makes no ECOA, HMDA, regulatory, lending, or production claim.

## Scope and version

`reason_code.reason_mapping` defines the current contract:

- `contract_version`: `demo-evidence-reason-mapping-v1`
- `catalog_version`: `demo-factor-catalog-v1`
- namespace: `demo_...` only

The contract consumes an adverse, non-zero attribution emitted by the bundled scorer labelled `demo_heuristic_not_calibrated`. It does not accept a trained model, a local HMDA research result, or an arbitrary `score_source`. `POST /decisions` remains bound to the injected demo decision service; loading data or running a research evaluation cannot replace that service or route a research model to it.

## Catalog dispositions

Every candidate factor resolves to exactly one status. Resolution is deterministic and never guesses a code.

| Status | Meaning | Explanation result |
| --- | --- | --- |
| `mapped` | A single supported demo mapping has a `demo_` code and display label. | The code, factor contribution, and reason-code evidence may be returned. |
| `unmapped` | The factor is not represented in this catalog, including an unknown factor. | Fail closed; no explanation response. |
| `ambiguous` | More than one demo interpretation could apply. | Fail closed; no explanation response. |
| `ineligible` | The factor is intentionally outside this demo explanation boundary. | Fail closed; no explanation response. |

The shipped catalog includes three mapped deterministic demo factors plus named test fixtures for the three refusal statuses. The fixtures are contract coverage, not claims about any real-world reason-code category.

## Evidence envelope

A successful `POST /decisions/explanation` response contains both the existing closed `evidence` set and `mapping_envelope`:

```json
{
  "case_id": "demo-001",
  "contract_version": "demo-evidence-reason-mapping-v1",
  "catalog_version": "demo-factor-catalog-v1",
  "mappings": [{
    "factor": "debt_to_income_ratio",
    "status": "mapped",
    "catalog_version": "demo-factor-catalog-v1",
    "demo_code": "demo_debt_to_income_risk_contribution",
    "label": "Debt-to-income ratio risk contribution"
  }],
  "explanation_evidence": {"case_id": "demo-001", "items": ["..."]}
}
```

The response validator requires the envelope case ID and nested evidence to equal the returned evidence, all envelope factors to be unique, every returned mapping to be `mapped`, and envelope codes to match the returned `reason_codes` exactly. A mapping record without a matching code and label, duplicated factor, mismatched catalog version, or any non-mapped status is rejected.

## Explicit local research path

The only end-to-end real-data research path is a deliberate local command with explicit paths:

```bash
PYTHONPATH=src python scripts/run_hmda_evaluation.py \
  --dataset-manifest /absolute/path/to/local-manifest.json \
  --output-dir /absolute/path/to/new-local-output
```

That command loads only the supplied local manifest, produces a non-authoritative observational research report, and does not call the API or this mapping contract. It must be run manually; neither the UI, `POST /decisions`, the explanation endpoint, CI, nor the deterministic fixture evaluation invokes it. This contract intentionally does not persist mapping artifacts or select review sets.
