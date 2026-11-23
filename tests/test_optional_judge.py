from __future__ import annotations

import json
from pathlib import Path

import pytest

from reason_code.evaluation import (
    DisabledJudge,
    JudgeMode,
    JudgeProvider,
    JudgeRequest,
    JudgeResponse,
    JudgeSettings,
    LlmOutputRejectedError,
    ManualExplanationGenerator,
    ManualExplanationJudge,
    ManualLlmClient,
    RecordedResponseJudge,
    build_local_judge,
    evidence_digest,
)
from reason_code.explanations import ExplanationEvidence

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recorded_synthetic_judge_response.json"
EVIDENCE = ExplanationEvidence.model_validate(
    {
        "case_id": "synthetic-faithful-001",
        "items": [
            {
                "evidence_id": "decision",
                "kind": "decision",
                "label": "Application requires human review",
            },
            {
                "evidence_id": "debt_to_income",
                "kind": "factor",
                "label": "Debt-to-income ratio",
                "value": "43%",
            },
            {
                "evidence_id": "reason_dti",
                "kind": "reason_code",
                "label": "Debt-to-income reason code",
            },
        ],
    }
)


def _fixture_response() -> JudgeResponse:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["status"].startswith("SYNTHETIC FIXTURE")
    assert fixture["metadata"] == {
        "synthetic": True,
        "non_authoritative": True,
        "network_used": False,
        "purpose": "Deterministic regression coverage for the optional recorded judge seam only.",
    }
    return JudgeResponse.model_validate(fixture["response"])


def _request(**changes: str) -> JudgeRequest:
    request = {
        "case_id": EVIDENCE.case_id,
        "evidence_digest": evidence_digest(EVIDENCE),
        "rubric_version": "synthetic-faithfulness-rubric-v1",
        "explanation_text": "Application requires human review.",
    }
    request.update(changes)
    return JudgeRequest.model_validate(request)


def test_recorded_response_matches_exact_case_digest_and_rubric() -> None:
    response = RecordedResponseJudge([_fixture_response()]).judge(_request())

    assert response == _fixture_response()
    assert response.accepted is True
    assert response.faithful is True


@pytest.mark.parametrize(
    ("change", "expected_finding"),
    [
        ("case_id", "No recorded response exists for this case_id."),
        ("evidence_digest", "Recorded response evidence_digest does not match request."),
        ("rubric_version", "Recorded response rubric_version does not match request."),
    ],
)
def test_recorded_response_judge_fails_closed_on_binding_mismatch(
    change: str, expected_finding: str
) -> None:
    replacement = "different-case" if change == "case_id" else "f" * 64
    if change == "rubric_version":
        replacement = "different-rubric"

    response = RecordedResponseJudge([_fixture_response()]).judge(_request(**{change: replacement}))

    assert response.accepted is False
    assert response.faithful is False
    assert response.findings == (expected_finding,)


def test_evidence_digest_is_stable_and_content_bound() -> None:
    assert evidence_digest(EVIDENCE) == _fixture_response().evidence_digest
    changed = EVIDENCE.model_copy(
        update={
            "items": (
                EVIDENCE.items[0],
                EVIDENCE.items[1].model_copy(update={"value": "44%"}),
                EVIDENCE.items[2],
            )
        }
    )
    assert evidence_digest(changed) != evidence_digest(EVIDENCE)


def test_manual_generator_accepts_only_deterministically_faithful_mocked_output() -> None:
    sent: list[tuple[str, bytes, object]] = []

    def transport(url: str, body: bytes, headers: object) -> bytes:
        sent.append((url, body, headers))
        return json.dumps(
            {
                "response": json.dumps(
                    {
                        "case_id": EVIDENCE.case_id,
                        "text": (
                            "Application requires human review. Debt-to-income ratio: 43%. "
                            "Debt-to-income reason code."
                        ),
                        "claims": [
                            {
                                "evidence_id": "decision",
                                "kind": "decision",
                                "subject": "Application requires human review",
                                "statement": "Application requires human review.",
                            },
                            {
                                "evidence_id": "debt_to_income",
                                "kind": "factor",
                                "subject": "Debt-to-income ratio",
                                "statement": "Debt-to-income ratio: 43%.",
                            },
                            {
                                "evidence_id": "reason_dti",
                                "kind": "reason_code",
                                "subject": "Debt-to-income reason code",
                                "statement": "Debt-to-income reason code.",
                            },
                        ],
                    }
                )
            }
        ).encode()

    client = ManualLlmClient(
        JudgeSettings(mode="manual", provider="ollama", model="local-test-model"), transport
    )
    generated = ManualExplanationGenerator(client).generate(EVIDENCE)

    assert generated.case_id == EVIDENCE.case_id
    assert len(sent) == 1
    prompt = json.loads(sent[0][1])["prompt"]
    assert json.loads(prompt.rsplit("\n", 1)[1]) == EVIDENCE.model_dump(mode="json")


def test_manual_generator_rejects_unfaithful_mocked_output() -> None:
    def transport(url: str, body: bytes, headers: object) -> bytes:
        unfaithful = {
            "case_id": "synthetic-faithful-001",
            "text": "Credit score: 710.",
            "claims": [
                {
                    "evidence_id": "decision",
                    "kind": "decision",
                    "subject": "Credit score",
                    "statement": "Credit score: 710.",
                }
            ],
        }
        return json.dumps({"response": json.dumps(unfaithful)}).encode()

    client = ManualLlmClient(
        JudgeSettings(mode="manual", provider="ollama", model="local-test-model"), transport
    )
    with pytest.raises(LlmOutputRejectedError, match="faithfulness"):
        ManualExplanationGenerator(client).generate(EVIDENCE)


def test_manual_judge_uses_mocked_response_and_preserves_request_bindings() -> None:
    def transport(url: str, body: bytes, headers: object) -> bytes:
        return b'{"response":"{\\"accepted\\":true,\\"faithful\\":true,\\"findings\\":[]}"}'

    client = ManualLlmClient(
        JudgeSettings(mode="manual", provider="ollama", model="local-test-model"), transport
    )
    response = ManualExplanationJudge(client).judge(_request())

    assert response.source == "manual"
    assert response.case_id == EVIDENCE.case_id
    assert response.evidence_digest == evidence_digest(EVIDENCE)


def test_recorded_judge_rejects_duplicate_case_responses() -> None:
    with pytest.raises(ValueError, match="unique case_id"):
        RecordedResponseJudge([_fixture_response(), _fixture_response()])


def test_default_settings_are_disabled_and_never_build_a_network_judge() -> None:
    settings = JudgeSettings.from_environment({"GROQ_API_KEY": "present-but-not-selected"})
    judge = build_local_judge(settings)

    assert settings.mode is JudgeMode.DISABLED
    assert settings.provider is JudgeProvider.RECORDED
    assert isinstance(judge, DisabledJudge)
    assert judge.judge(_request()).accepted is False


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:11434/api/generate", "http://localhost:11434"]
)
def test_manual_ollama_only_accepts_loopback_http_endpoint(endpoint: str) -> None:
    settings = JudgeSettings(mode="manual", provider="ollama", endpoint=endpoint)

    assert settings.resolved_endpoint == endpoint
    with pytest.raises(ValueError, match="explicit host-controlled invocation"):
        build_local_judge(settings)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434/api/generate",
        "http://192.168.1.8:11434/api/generate",
        "http://example.test/api/generate",
        "http://user:password@127.0.0.1:11434/api/generate",
        "http://127.0.0.1:11434/api/generate?fallback=1",
    ],
)
def test_ollama_rejects_non_loopback_or_arbitrary_urls(endpoint: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        JudgeSettings(mode="manual", provider="ollama", endpoint=endpoint)


@pytest.mark.parametrize(
    ("provider", "key_name", "endpoint"),
    [
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions"),
        ("gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta"),
    ],
)
def test_manual_remote_providers_require_explicit_selection_and_key(
    provider: str, key_name: str, endpoint: str
) -> None:
    environment = {"JUDGE_MODE": "manual", "JUDGE_PROVIDER": provider}
    with pytest.raises(ValueError, match=key_name):
        JudgeSettings.from_environment(environment)

    settings = JudgeSettings.from_environment({**environment, key_name: "test-key"})

    assert settings.resolved_endpoint == endpoint
    assert settings.api_key_from({key_name: "test-key"}) == "test-key"
    with pytest.raises(ValueError, match="explicit host-controlled invocation"):
        build_local_judge(settings)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "azure"])
def test_forbidden_providers_are_rejected_without_fallback(provider: str) -> None:
    with pytest.raises(ValueError, match="not permitted"):
        JudgeSettings.from_environment({"JUDGE_PROVIDER": provider})


def test_recorded_mode_refuses_provider_or_endpoint_fallbacks() -> None:
    with pytest.raises(ValueError, match="recorded mode"):
        JudgeSettings(mode="recorded", provider="ollama")
    with pytest.raises(ValueError, match="disabled mode"):
        JudgeSettings(mode="disabled", endpoint="http://127.0.0.1:11434")
    with pytest.raises(ValueError, match="requires explicit ollama"):
        JudgeSettings(mode="manual", provider="recorded")
