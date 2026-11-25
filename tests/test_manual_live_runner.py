from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reason_code.evaluation import JudgeSettings
from reason_code.evaluation.live_runner import (
    LIVE_FIXTURE_CASE_ID,
    LIVE_REQUEST_CAP,
    minimum_output_token_cap,
    run_live_experiment,
)
from reason_code.evaluation.manual import MAX_RESPONSE_BYTES


def _faithful_ollama_response() -> bytes:
    return json.dumps(
        {
            "response": json.dumps(
                {
                    "case_id": LIVE_FIXTURE_CASE_ID,
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


def test_live_experiment_makes_one_fake_request_and_records_safe_metadata() -> None:
    calls: list[tuple[str, bytes, object, float]] = []

    def transport(url: str, body: bytes, headers: object, timeout_seconds: float) -> bytes:
        calls.append((url, body, headers, timeout_seconds))
        return _faithful_ollama_response()

    report = run_live_experiment(
        JudgeSettings(
            mode="manual", provider="ollama", model="library/llama3.2:3b", timeout_seconds=5
        ),
        transport=transport,
    )

    assert report["status"] == "pass"
    assert report["metadata"]["non_authoritative"] is True
    assert report["metadata"]["n"] == 1
    assert report["metadata"]["request_cap"] == LIVE_REQUEST_CAP == 1
    assert report["metadata"]["output_token_cap"] >= report["metadata"]["minimum_output_token_cap"]
    assert len(calls) == 1
    assert calls[0][3] == 5
    assert (
        json.loads(calls[0][1])["options"]["num_predict"] == report["metadata"]["output_token_cap"]
    )
    assert "output_sha256" in report
    assert "response" not in report


@pytest.mark.parametrize(
    ("response", "cause"),
    [
        (b"not-json", "malformed_output"),
        (
            json.dumps({"response": json.dumps({"case_id": LIVE_FIXTURE_CASE_ID})}).encode(),
            "schema_rejection",
        ),
        (
            json.dumps(
                {
                    "response": json.dumps(
                        {
                            "case_id": LIVE_FIXTURE_CASE_ID,
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
                    )
                }
            ).encode(),
            "unfaithfulness",
        ),
    ],
)
def test_live_experiment_distinguishes_output_rejection_causes(response: bytes, cause: str) -> None:
    report = run_live_experiment(
        JudgeSettings(mode="manual", provider="ollama", model="local-test-model"),
        transport=lambda url, body, headers, timeout: response,
    )

    assert report["status"] == "fail"
    assert report["rejection_cause"] == cause
    assert report["metadata"]["n"] == 1
    assert report["metadata"]["non_authoritative"] is True


def test_live_experiment_classifies_provider_failure_and_bounds_response_bytes() -> None:
    report = run_live_experiment(
        JudgeSettings(mode="manual", provider="ollama", model="local-test-model"),
        transport=lambda url, body, headers, timeout: b"x" * (MAX_RESPONSE_BYTES + 1),
    )
    assert report["rejection_cause"] == "provider_failure"


def test_live_experiment_rejects_insufficient_evidence_dependent_cap_before_transport() -> None:
    called = False

    def transport(url: str, body: bytes, headers: object, timeout: float) -> bytes:
        nonlocal called
        called = True
        return _faithful_ollama_response()

    settings = JudgeSettings(
        mode="manual", provider="ollama", model="local-test-model", output_token_cap=1
    )
    report = run_live_experiment(settings, transport=transport)
    assert report["rejection_cause"] == "configuration"
    assert called is False
    assert report["metadata"]["minimum_output_token_cap"] > 1


def test_live_experiment_rejects_absent_remote_key_before_transport() -> None:
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        run_live_experiment(
            JudgeSettings(mode="manual", provider="groq", model="free-model"),
            transport=lambda url, body, headers, timeout: pytest.fail(
                "transport must not be called"
            ),
            environ={},
        )


def test_manual_client_allows_one_transport_request_structurally() -> None:
    from reason_code.evaluation.manual import LlmProviderRequestError, ManualLlmClient

    client = ManualLlmClient(
        JudgeSettings(mode="manual", provider="ollama", model="local-test-model"),
        transport=lambda url, body, headers, timeout: b'{"response":"{}"}',
    )
    client.complete_json("first")
    with pytest.raises(LlmProviderRequestError, match="exactly one"):
        client.complete_json("second")


def test_minimum_cap_is_bounded_by_supported_settings_maximum() -> None:
    from reason_code.evaluation.live_runner import _fixture_evidence

    evidence, _ = _fixture_evidence()
    assert 1 < minimum_output_token_cap(evidence) <= 1024


@pytest.mark.parametrize(
    ("provider", "environment", "payload_path"),
    [
        ("ollama", {}, ("options", "num_predict")),
        ("groq", {"GROQ_API_KEY": "test-key"}, ("max_tokens",)),
        ("gemini", {"GEMINI_API_KEY": "test-key"}, ("generationConfig", "maxOutputTokens")),
    ],
)
def test_provider_specific_token_caps_and_timeout_are_plumbed_to_fake_transport(
    provider: str, environment: dict[str, str], payload_path: tuple[str, ...]
) -> None:
    from reason_code.evaluation.manual import LlmMalformedResponseError, ManualLlmClient

    observed: list[float] = []

    def transport(url: str, body: bytes, headers: object, timeout: float) -> bytes:
        observed.append(timeout)
        return b"{}"

    # Request construction is the unit under test; Gemini's fake response is not parsed.
    client = ManualLlmClient(
        JudgeSettings(
            mode="manual",
            provider=provider,
            model="library/model:tag",
            output_token_cap=768,
            timeout_seconds=7,
        ),
        transport=transport,
        environ=environment,
    )
    _, body, _ = client._request_parts("prompt")
    payload: object = json.loads(body)
    for key in payload_path:
        assert isinstance(payload, dict)
        payload = payload[key]
    assert payload == 768
    with pytest.raises(LlmMalformedResponseError):
        client.complete_json("prompt")
    assert observed == [7]


def test_blank_token_and_timeout_environment_values_use_safe_defaults() -> None:
    settings = JudgeSettings.from_environment(
        {
            "JUDGE_MODE": "manual",
            "JUDGE_PROVIDER": "ollama",
            "JUDGE_MODEL": "llama3.2:3b",
            "JUDGE_OUTPUT_TOKEN_CAP": "",
            "JUDGE_TIMEOUT_SECONDS": "",
        }
    )
    assert settings.output_token_cap == 768
    assert settings.timeout_seconds == 15


def test_cli_requires_confirmation_and_emits_json_only_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reason_code.evaluation.live_run import main

    monkeypatch.setattr("sys.argv", ["live-run", "--provider", "ollama", "--model", "llama3.2:3b"])
    assert main() == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["rejection_cause"] == "configuration"
    assert report["metadata"]["n"] == 1
    assert captured.out.count("\n") == 1


def test_cli_success_stdout_is_one_json_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reason_code.evaluation.live_run import main

    monkeypatch.setattr(
        "reason_code.evaluation.live_run.run_live_experiment",
        lambda settings: {
            "status": "pass",
            "metadata": {"n": 1, "non_authoritative": True},
            "verdict": {},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["live-run", "--confirm-live-request", "--provider", "ollama", "--model", "llama3.2:3b"],
    )
    assert main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "pass"
    assert captured.out.count("\n") == 1


def test_transport_uses_direct_loopback_connection_despite_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reason_code.evaluation.manual import _default_transport

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("http_proxy", "http://192.0.2.1:9")
        monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:9")
        port = server.server_address[1]
        assert _default_transport(f"http://127.0.0.1:{port}/", b"{}", {}, 1) == b"{}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_transport_rejects_redirects_and_enforces_byte_and_deadline_limits() -> None:
    from reason_code.evaluation.manual import LlmProviderRequestError, _default_transport

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/ok")
                self.end_headers()
                return
            if self.path == "/large":
                body = b"x" * (MAX_RESPONSE_BYTES + 1)
            elif self.path == "/slow":
                body = b"x" * 8
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for byte in body:
                    self.wfile.write(bytes((byte,)))
                    self.wfile.flush()
                    time.sleep(0.02)
                return
            else:
                body = b"{}"
            self.send_response(200)
            if self.path == "/connection-close":
                self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for path in ("/redirect", "/large"):
            with pytest.raises(LlmProviderRequestError):
                _default_transport(f"{base_url}{path}", b"{}", {}, 1)
        assert _default_transport(f"{base_url}/connection-close", b"{}", {}, 1) == b"{}"
        with pytest.raises(LlmProviderRequestError):
            _default_transport(f"{base_url}/slow", b"{}", {}, 0.05)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_manual_client_propagates_network_guard_and_programming_errors() -> None:
    from reason_code.evaluation.manual import ManualLlmClient

    client = ManualLlmClient(
        JudgeSettings(mode="manual", provider="ollama", model="local-test-model"),
        transport=lambda url, body, headers, timeout: (_ for _ in ()).throw(
            AssertionError("network guard assertion")
        ),
    )
    with pytest.raises(AssertionError, match="network guard assertion"):
        client.complete_json("prompt")


def test_fixture_integrity_failure_is_redacted_and_not_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import reason_code.evaluation.live_runner as live_runner
    from reason_code.evaluation.runner import EvaluationError

    monkeypatch.setattr(
        live_runner,
        "_fixture_evidence",
        lambda: (_ for _ in ()).throw(EvaluationError("fixture SECRET_VALUE")),
    )
    report = run_live_experiment(JudgeSettings(mode="manual", provider="ollama", model="local"))
    serialized = json.dumps(report)
    assert report["rejection_cause"] == "fixture_integrity"
    assert "SECRET_VALUE" not in serialized
    assert "fixture SECRET_VALUE" not in serialized


def test_cli_redacts_unexpected_configuration_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reason_code.evaluation.live_run import main

    monkeypatch.setattr(
        "reason_code.evaluation.live_run.run_live_experiment",
        lambda settings: (_ for _ in ()).throw(ValueError("api-key=SECRET_VALUE")),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["live-run", "--confirm-live-request", "--provider", "ollama", "--model", "local"],
    )
    assert main() == 1
    captured = capsys.readouterr()
    assert "SECRET_VALUE" not in captured.out
    assert "SECRET_VALUE" not in captured.err
