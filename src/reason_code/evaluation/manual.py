"""Explicit, manual-only HTTP adapters for evidence-constrained LLM evaluation.

These adapters are intentionally not wired into the application or local judge builder.
A host must select a permitted provider, supply a model and key where applicable, and
call them directly. Tests inject a transport and never use the default HTTP transport.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from reason_code.explanations import (
    ExplanationEvidence,
    GeneratedExplanation,
    grade_faithfulness,
    grade_prose_evidence,
)

from .config import JudgeMode, JudgeProvider, JudgeSettings
from .digest import evidence_digest
from .models import JudgeRequest, JudgeResponse

Transport = Callable[[str, bytes, Mapping[str, str]], bytes]


class LlmOutputRejectedError(ValueError):
    """The manually requested model output did not satisfy deterministic boundaries."""


def _default_transport(url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    with urlopen(request, timeout=30) as response:  # noqa: S310 - endpoint is policy-validated
        return response.read()


class ManualLlmClient:
    """Provider-policy-checked, one-shot JSON client with no fallback behavior."""

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

    def complete_json(self, prompt: str) -> dict[str, Any]:
        url, body, headers = self._request_parts(prompt)
        try:
            raw = self._transport(url, body, headers)
            payload = json.loads(raw)
            content = self._content_from(payload)
            decoded = json.loads(content)
        except (IndexError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LlmOutputRejectedError("manual LLM returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise LlmOutputRejectedError("manual LLM JSON response must be an object")
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
                "response_format": {"type": "json_object"},
            }
            return endpoint, json.dumps(payload).encode(), headers
        if self._settings.provider is JudgeProvider.GEMINI:
            if key is None:
                raise ValueError("Gemini calls require an API key")
            headers["x-goog-api-key"] = key
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
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
    """Generate only from supplied evidence and reject anything the deterministic grader rejects."""

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
        except (ValueError, TypeError) as exc:
            raise LlmOutputRejectedError("manual LLM explanation failed typed validation") from exc
        prose_verdict = grade_prose_evidence(explanation, evidence)
        structural_verdict = grade_faithfulness(explanation, evidence)
        if not prose_verdict.faithful or not structural_verdict.faithful:
            raise LlmOutputRejectedError(
                "manual LLM explanation failed deterministic evidence validation"
            )
        return explanation


class ManualExplanationJudge:
    """Manual LLM-as-judge adapter, bound to caller-provided request identifiers."""

    def __init__(self, client: ManualLlmClient) -> None:
        self._client = client

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        prompt = (
            "Return only a JSON object with accepted, faithful, and findings. "
            "This is non-authoritative. Do not change these bindings: "
            + json.dumps(request.model_dump(), sort_keys=True)
        )
        try:
            result = self._client.complete_json(prompt)
            response = JudgeResponse.model_validate(
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
        except (KeyError, TypeError, ValueError) as exc:
            raise LlmOutputRejectedError(
                "manual LLM judge response failed typed validation"
            ) from exc
        return response


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
