"""Manual-only, bounded one-request adapters for evidence-constrained LLM evaluation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from pydantic import ValidationError

from reason_code.explanations import (
    ExplanationEvidence,
    GeneratedExplanation,
    grade_faithfulness,
    grade_prose_evidence,
)

from .config import JudgeMode, JudgeProvider, JudgeSettings
from .digest import evidence_digest
from .models import JudgeRequest, JudgeResponse

MAX_RESPONSE_BYTES = 65_536
Transport = Callable[[str, bytes, Mapping[str, str], float], bytes]


class LlmOutputRejectedError(ValueError):
    """Base class for output rejected without retaining provider content."""


class LlmMalformedResponseError(LlmOutputRejectedError):
    """Provider envelope or JSON syntax was malformed."""


class LlmSchemaRejectedError(LlmOutputRejectedError):
    """Parsed JSON did not satisfy the required typed output schema."""


class LlmUnfaithfulOutputError(LlmOutputRejectedError):
    """Typed output failed deterministic evidence validation."""


class LlmProviderRequestError(RuntimeError):
    """A provider transport failed, timed out, or exceeded local byte limits."""


class _RejectRedirects(HTTPRedirectHandler):
    """Treat every HTTP redirect as a transport failure, never a follow-up request."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


class _DeadlineHandler:
    """Keep urllib's live connection available while incrementally reading its response."""

    def do_open(self, http_class: object, req: Request, **http_conn_args: object) -> object:
        host = req.host
        if not host:
            raise URLError("no host given")
        connection = http_class(host, timeout=req.timeout, **http_conn_args)
        connection.set_debuglevel(self._debuglevel)
        headers = dict(req.unredirected_hdrs)
        headers.update({key: value for key, value in req.headers.items() if key not in headers})
        headers["Connection"] = "close"
        headers = {name.title(): value for name, value in headers.items()}
        try:
            connection.request(
                req.get_method(),
                req.selector,
                req.data,
                headers,
                encode_chunked=req.has_header("Transfer-encoding"),
            )
            # ``getresponse()`` calls ``connection.close()`` for HTTP/1.0 and peers that
            # honor our Connection: close header. Suppress only that internal teardown:
            # the response owns this one socket until the bounded read loop closes it.
            live_socket = connection.sock
            connection.close = lambda: None
            try:
                response = connection.getresponse()
            finally:
                del connection.close
        except BaseException:
            connection.close()
            raise
        # Keep the real socket for incremental deadline-bounded reads. The caller closes it.
        response._reason_code_socket = live_socket
        response.url = req.get_full_url()
        response.msg = response.reason
        return response


class _DeadlineHTTPHandler(_DeadlineHandler, HTTPHandler):
    pass


class _DeadlineHTTPSHandler(_DeadlineHandler, HTTPSHandler):
    pass


def _transport_opener() -> object:
    """Construct an opener that ignores proxy environment variables and redirects."""
    return build_opener(
        ProxyHandler({}), _RejectRedirects(), _DeadlineHTTPHandler(), _DeadlineHTTPSHandler()
    )


def _response_socket(response: object) -> object:
    """Return urllib's live response socket or refuse a transport we cannot deadline-bound."""
    socket_handle = getattr(response, "_reason_code_socket", None)
    if socket_handle is None or not callable(getattr(socket_handle, "settimeout", None)):
        raise LlmProviderRequestError("provider transport cannot enforce deadline")
    return socket_handle


def _default_transport(
    url: str, body: bytes, headers: Mapping[str, str], timeout_seconds: float
) -> bytes:
    """Send exactly one bounded request under one wall-clock deadline."""
    deadline = time.monotonic() + timeout_seconds
    request = Request(url, data=body, headers=dict(headers), method="POST")
    chunks: list[bytes] = []
    received = 0
    try:
        opener = _transport_opener()
        with opener.open(request, timeout=max(0.001, deadline - time.monotonic())) as response:
            socket_handle = _response_socket(response)
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("deadline elapsed")
                    socket_handle.settimeout(remaining)
                    # ``read`` buffers until it fills its requested size, so a slow peer can
                    # reset a per-recv socket timeout forever. ``read1`` returns after one read.
                    chunk = response.read1(min(8192, MAX_RESPONSE_BYTES + 1 - received))
                    if time.monotonic() > deadline:
                        raise TimeoutError("deadline elapsed")
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > MAX_RESPONSE_BYTES:
                        raise LlmProviderRequestError("provider response exceeded local byte limit")
                    chunks.append(chunk)
            finally:
                socket_handle.close()
    except LlmProviderRequestError:
        raise
    except (OSError, TimeoutError, URLError) as exc:
        raise LlmProviderRequestError("provider request failed") from exc
    return b"".join(chunks)


class ManualLlmClient:
    """Provider-policy-checked client that structurally permits exactly one request."""

    def __init__(
        self,
        settings: JudgeSettings,
        transport: Transport | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        if settings.mode is not JudgeMode.MANUAL:
            raise ValueError("manual LLM clients require manual judge mode")
        if settings.provider is JudgeProvider.RECORDED:
            raise ValueError("recorded provider cannot make manual LLM calls")
        if not settings.model:
            raise ValueError("manual LLM calls require an explicit JUDGE_MODEL")
        endpoint = settings.resolved_endpoint
        if endpoint is None:
            raise ValueError("manual LLM calls require a configured endpoint")
        self._settings = settings
        self._endpoint = endpoint
        self._api_key = settings.api_key_from(environ)
        self._transport = transport or _default_transport
        self._request_used = False

    def complete_json(self, prompt: str) -> dict[str, Any]:
        """Perform the sole transport invocation and parse a provider JSON envelope."""
        if self._request_used:
            raise LlmProviderRequestError("manual client permits exactly one transport request")
        self._request_used = True
        url, body, headers = self._request_parts(prompt)
        try:
            raw = self._transport(url, body, headers, self._settings.timeout_seconds)
        except LlmProviderRequestError:
            raise
        except (OSError, TimeoutError, URLError) as exc:
            raise LlmProviderRequestError("provider request failed") from exc
        if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
            raise LlmProviderRequestError("provider response exceeded local byte limit")
        try:
            payload = json.loads(raw)
            content = self._content_from(payload)
            decoded = json.loads(content)
        except (IndexError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LlmMalformedResponseError("manual LLM returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise LlmMalformedResponseError("manual LLM JSON response must be an object")
        return decoded

    def _request_parts(self, prompt: str) -> tuple[str, bytes, dict[str, str]]:
        endpoint = self._endpoint
        headers = {"content-type": "application/json"}
        if self._settings.provider is JudgeProvider.OLLAMA:
            payload = {
                "model": self._settings.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"num_predict": self._settings.output_token_cap},
            }
            return endpoint, json.dumps(payload).encode(), headers
        key = self._api_key
        if self._settings.provider is JudgeProvider.GROQ:
            if key is None:
                raise ValueError("Groq calls require an API key")
            headers["authorization"] = f"Bearer {key}"
            payload = {
                "model": self._settings.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self._settings.output_token_cap,
                "response_format": {"type": "json_object"},
            }
            return endpoint, json.dumps(payload).encode(), headers
        if self._settings.provider is JudgeProvider.GEMINI:
            if key is None:
                raise ValueError("Gemini calls require an API key")
            headers["x-goog-api-key"] = key
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0,
                    "maxOutputTokens": self._settings.output_token_cap,
                },
            }
            model = quote(self._settings.model or "", safe="")
            return (
                f"{endpoint}/models/{model}:generateContent",
                json.dumps(payload).encode(),
                headers,
            )
        raise ValueError("manual LLM provider is not supported")

    def _content_from(self, payload: Mapping[str, Any]) -> str:
        if self._settings.provider is JudgeProvider.OLLAMA:
            return str(payload["response"])
        if self._settings.provider is JudgeProvider.GROQ:
            return str(payload["choices"][0]["message"]["content"])
        return str(payload["candidates"][0]["content"]["parts"][0]["text"])


class ManualExplanationGenerator:
    """Generate only from supplied evidence and reject deterministic failures."""

    def __init__(self, client: ManualLlmClient) -> None:
        self._client = client

    def generate(self, evidence: ExplanationEvidence) -> GeneratedExplanation:
        allowed_evidence = evidence.model_dump(mode="json")
        prompt = (
            "Return only a JSON object matching GeneratedExplanation with case_id, text, "
            "and claims. Every claim must cite one supplied evidence_id, use its exact kind "
            "and label, and render only that item. Do not add facts. Use exactly this "
            "closed evidence set:\n" + json.dumps(allowed_evidence, sort_keys=True)
        )
        try:
            explanation = GeneratedExplanation.model_validate(self._client.complete_json(prompt))
        except ValidationError as exc:
            raise LlmSchemaRejectedError("manual LLM explanation failed typed validation") from exc
        prose_verdict = grade_prose_evidence(explanation, evidence)
        structural_verdict = grade_faithfulness(explanation, evidence)
        if not prose_verdict.faithful or not structural_verdict.faithful:
            raise LlmUnfaithfulOutputError(
                "manual LLM explanation failed deterministic evidence validation"
            )
        return explanation


class ManualExplanationJudge:
    """Manual LLM-as-judge adapter, bound to caller-provided request identifiers."""

    def __init__(self, client: ManualLlmClient) -> None:
        self._client = client

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        prompt = (
            "Return only JSON with accepted, faithful, and findings. Non-authoritative: "
            + json.dumps(request.model_dump(), sort_keys=True)
        )
        try:
            result = self._client.complete_json(prompt)
            return JudgeResponse.model_validate(
                {
                    "case_id": request.case_id,
                    "evidence_digest": request.evidence_digest,
                    "rubric_version": request.rubric_version,
                    "accepted": result["accepted"],
                    "faithful": result["faithful"],
                    "findings": result.get("findings", []),
                    "source": "manual",
                    "synthetic": False,
                }
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise LlmSchemaRejectedError(
                "manual LLM judge response failed typed validation"
            ) from exc


def request_for(
    evidence: ExplanationEvidence, rubric_version: str, explanation_text: str
) -> JudgeRequest:
    """Bind a manual judge request to exactly the evidence passed to the generator."""
    return JudgeRequest(
        case_id=evidence.case_id,
        evidence_digest=evidence_digest(evidence),
        rubric_version=rubric_version,
        explanation_text=explanation_text,
    )
