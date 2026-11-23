# No-cost operating policy

Reason Code must be buildable, testable, and reviewable without paid services, credentials, or automatic billable work.

## Default path: zero spend

- Local development and the test suite use only synthetic fixtures, deterministic programmatic grading, and local Python dependencies.
- Pull-request CI must be hermetic: no network access, no model/API calls, no external dataset download, and no cloud database or hosted inference dependency.
- The repository does not contain an API key, billing credential, paid-model default, or automatic deployment workflow.
- The public repository uses GitHub-hosted CI only for the deterministic test/lint path. No self-hosted runner, marketplace action with a paid plan, or scheduled workflow is part of the default path.

## Explicit opt-in only

The following are **not** part of normal development or PR CI. They require a deliberate local command, a user-provided credential, and a documented cost check before they run:

1. downloading an HMDA data slice;
2. calling an LLM generator or judge;
3. provisioning any cloud database, model endpoint, observability service, or deployment;
4. recording or hosting a public demo.

If an LLM judge is later evaluated, its recorded-response fixture remains the test default. The deterministic programmatic faithfulness grader is the blocking quality gate.

## Cost-review checklist

Before adding a dependency or workflow, answer all four questions in the pull request or commit message:

1. Does it make a network request automatically?
2. Can it create a paid cloud/API/GitHub charge?
3. Is there a free local or fixture-backed alternative for CI?
4. Has it been made opt-in, with credentials excluded from the repository?

If any answer is unclear, do not enable it.
