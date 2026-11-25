# Human evaluation protocol (proposed; not human-reviewed)

## Status and hard boundary

This is a **proposed, test-only protocol**. It is not a completed human study, a reported quality result, a lending assessment, or a regulatory claim. The exporter makes no network request, LLM call, FastAPI request, model inference, training, or model/reason-code mapping change. It reads only two explicitly named local files: a validated dataset manifest and a JSON artifact exported from the evaluator's held-out test predictions.

The implementation is `reason_code.evaluation.human_review`. Do not represent synthetic output from it as human-reviewed evidence.

## Fixed proposed 75-packet design

`select_resume_gate_packet(dataset_manifest_path, model_artifact_path)` produces exactly 75 blinded cases from the held-out `test` partition.

### Natural cases: 50

There are exactly five natural cases in every 2 observed-outcome × 5 empirical probability-rank stratum:

| Observed outcome | `q1` | `q2` | `q3` | `q4` | `q5` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Label `0` | 5 | 5 | 5 | 5 | 5 |
| Label `1` | 5 | 5 | 5 | 5 | 5 |

`q1` through `q5` are deterministic empirical quintile **rank bands** built from the bound artifact's full held-out prediction set. They are not fixed score intervals, calibration claims, or decision-policy thresholds. Predictions sort by calibrated probability; exact probability ties use a hash of the bound manifest/artifact fingerprints and record ID. This keeps selection valid when all genuine probabilities occupy a compressed numeric range.

### Controlled explanation packets: 25

- **16 invalid controls:** two for each visible contradiction type: annual income plus one, requested amount plus one, debt-to-income ratio plus one, annual income substituted with requested amount, requested amount substituted with annual income, debt-to-income substituted with annual income, annual income substituted with debt-to-income, and requested amount substituted with debt-to-income. Every invalid statement is visibly inconsistent with the displayed facts; no hidden evidence IDs, claim kinds, or trace-only error labels are used as human-review controls.
- **9 valid candidates:** three each from rank regions `q1`, `q3`, and `q5`. These mean lower-, middle-, and higher-ranked *within this exact held-out prediction artifact*, not low/medium/high risk and not a policy.

Controls have the same blinded public shape as natural cases. Control type and expected verdict remain in private linkage only. A source record may appear in **only one** natural or control case; selection reserves valid controls first and then selects invalid controls only from unconsumed test IDs. Construction rejects both source-ID reuse and identical visible facts/statement content with conflicting expected verdicts.

## Exact evaluator artifact contract

The dataset manifest is explicit and uses the existing local-only checksum-validating loader. The evaluator must export its **exact post-calibration held-out test predictions** to an explicit local JSON file:

```json
{
  "artifact_version": "resume-gate-calibration-v2",
  "dataset_manifest_sha256": "<raw SHA-256 of the exact dataset manifest file>",
  "evaluator_version": "<non-empty evaluator implementation/version identifier>",
  "split": "test",
  "predictions": [
    {"record_id": "<test record ID>", "calibrated_probability": 0.0573}
  ],
  "test_predictions_sha256": "<SHA-256 of canonical, record_id-sorted predictions>"
}
```

Required evaluator changes:

1. At the point where the evaluator has produced final **calibrated** probabilities for its chronological `test` split, emit exactly one record per test record with the original manifest `record_id` and finite probability in `[0, 1]`.
2. Sort those two-key objects by `record_id`, serialize with JSON `sort_keys=True`, `separators=(",", ":")`, and ASCII encoding, then write its SHA-256 to `test_predictions_sha256`.
3. Include the raw SHA-256 of the supplied dataset manifest, a stable evaluator version identifier, and literal `split: "test"`.
4. Do not include train/validation predictions, decision outcomes, reason-code mappings, source rows, or model objects in this artifact.

The exporter requires an exact one-to-one match between artifact IDs and loaded held-out test IDs, validates the prediction hash, then derives the rank bands. It fails closed on malformed/boundary-mismatched artifacts and any natural, valid-control, or invalid-control stratum shortage; it never emits a partial packet.

The local CLI is export-only:

```bash
python -m reason_code.evaluation.human_review \
  --dataset-manifest /absolute/path/dataset-manifest.json \
  --model-artifact /absolute/path/test-predictions.json \
  --packet /absolute/path/blinded-packet.json \
  --private-linkage /absolute/path/private-linkage.json
```

## Blinding and linkage separation

The public packet contains only an opaque case ID, displayed facts, and a candidate statement. It excludes source IDs, labels/outcomes, calibrated probabilities, rank bands, expected verdicts, control types, failure codes, and input hashes.

Private linkage has that metadata and is separately exportable. Store it outside reviewer-accessible locations; do not use it during independent review. `export_review_artifacts` refuses to write public packet and private linkage to one path.

## Annotation, adjudication, freeze, and report

Each blinded annotation contains the fixed annotation version, packet ID, opaque case ID, reviewer ID, `accept`/`reject`/`uncertain` verdict, and non-empty single-line notes. `validate_independent_reviews` requires exactly one annotation for every packet case from each independent reviewer.

After review closes, an adjudicator who was not a reviewer must provide one final verdict for each disagreement and none for agreement. `freeze_review` records canonical SHA-256 hashes of packet, validated reviews, and adjudications with frozen counts. `review_report` verifies the packet hash before post-freeze aggregate counts; it may use private linkage only after that check.

## Limitations

- This is a protocol proposal with synthetic tests, not human-reviewed study evidence.
- Empirical rank bands provide sampling balance only; they do not validate calibration, establish score meaning, or define lending policy.
- The artifact is evaluator output; the exporter neither performs inference nor validates model calibration.
- Controlled invalid cases exercise visible fact/statement contradictions, not a natural model-error rate.
- Opaque IDs hide source linkage, but displayed facts remain subject to source-data handling controls.
