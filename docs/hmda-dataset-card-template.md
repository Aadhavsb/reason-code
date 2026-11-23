# HMDA Dataset Card Template

> Template only. It documents a future acquisition; this repository does **not** include or claim to have acquired HMDA data. Complete every bracketed field only after local acquisition and verification.

## Dataset identity

- **Dataset name:** [e.g., HMDA Loan/Application Register]
- **Source publisher:** [publisher name]
- **Source URL:** [exact HTTPS URL]
- **Source version or release identifier:** [publisher-provided identifier]
- **Local acquisition status:** `not_acquired` | `acquired_locally`
- **Download date (UTC):** [YYYY-MM-DD]
- **Source version hash:** [SHA-256 of the downloaded source or publisher-provided version artifact]
- **Manifest hash:** [output of `python3 -m reason_code.provenance hash <manifest.json>`]

## Intended use and limits

- **Project use:** offline decision-support demonstration and explanation-evaluation research.
- **Not for:** production underwriting, lending advice, eligibility decisions, or deployment.
- **Interpretation limit:** model outputs describe associations in the chosen sample; they do not establish causality.
- **Fairness limit:** any demographic analysis is descriptive by split and sample only. It is not a fair-lending determination or causal fairness claim.

## Acquisition and filtering

- **Acquisition method:** [manual download / documented API request; do not include credentials]
- **Exact filter object:** [copy the JSON `filter` object from the provenance manifest]
- **Geography:** [included values or `not filtered`]
- **Activity year(s):** [included values]
- **Action-taken / population filters:** [included values]
- **Rows before filtering:** [measured count, or `not yet measured`]
- **Rows after filtering:** [measured count, or `not yet measured`]
- **Exclusions and rationale:** [each exclusion with its reason]

## Fields and transformations

| Raw field | Use | Transformation / missing-value handling | Included in model? |
| --- | --- | --- | --- |
| [field] | [purpose] | [documented operation] | [yes/no] |

List every field actually used. Do not include protected or sensitive fields in a model merely because they appear in a source extract; document any descriptive-only use separately.

## Split and evaluation record

- **Record date field:** [field name]
- **Chronological split specification:** [serialized `ChronologicalSplitSpec.as_dict()`]
- **Training period:** [measured date range]
- **Validation period:** [measured date range]
- **Test period:** [measured date range]
- **Evaluation-set version:** [identifier, or `not yet created`]
- **Metrics:** [measured values only; name split and assumptions]

## Provenance manifest procedure

Create a local JSON file containing exactly these required keys:

```json
{
  "schema_version": "1.0",
  "source_url": "https://example.gov/hmda/source.csv",
  "filter": {"activity_year": 2025},
  "download_date": "2026-11-23",
  "fields": ["activity_year", "action_taken"],
  "source_version_hash": "<64-character SHA-256 hex digest>"
}
```

The URL and date above are placeholders, not a record of an HMDA download. Validate locally without any network request:

```bash
PYTHONPATH=src python3 -m reason_code.provenance validate path/to/manifest.json
PYTHONPATH=src python3 -m reason_code.provenance hash path/to/manifest.json
```

A successful validation confirms manifest structure only. It does not confirm acquisition, publisher authenticity, file integrity, data quality, representativeness, legal compliance, model validity, or any fairness or causal conclusion.
