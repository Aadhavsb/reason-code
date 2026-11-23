"""Strict, opt-in configuration for optional explanation judges."""

from __future__ import annotations

import os
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JudgeMode(StrEnum):
    DISABLED = "disabled"
    RECORDED = "recorded"
    MANUAL = "manual"


class JudgeProvider(StrEnum):
    RECORDED = "recorded"
    OLLAMA = "ollama"
    GROQ = "groq"
    GEMINI = "gemini"


_FORBIDDEN_PROVIDER_NAMES = frozenset({"openai", "anthropic", "azure"})
_OLLAMA_DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/generate"
_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


class JudgeSettings(BaseModel):
    """Configuration that refuses paid, remote-default, and fallback execution.

    The default is disabled. Recorded mode stays local and deterministic; manual mode
    is the only mode in which a caller may choose one of the explicit providers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: JudgeMode = JudgeMode.DISABLED
    provider: JudgeProvider = JudgeProvider.RECORDED
    endpoint: str | None = None
    model: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> JudgeSettings:
        if self.provider.value in _FORBIDDEN_PROVIDER_NAMES:
            raise ValueError("OpenAI, Anthropic, and Azure judge providers are not permitted")
        if self.mode is JudgeMode.DISABLED:
            if self.provider is not JudgeProvider.RECORDED or self.endpoint is not None:
                raise ValueError(
                    "disabled mode only permits the recorded provider without an endpoint"
                )
            return self
        if self.mode is JudgeMode.RECORDED:
            if self.provider is not JudgeProvider.RECORDED or self.endpoint is not None:
                raise ValueError(
                    "recorded mode only permits the recorded provider without an endpoint"
                )
            return self
        if self.provider is JudgeProvider.RECORDED:
            raise ValueError(
                "manual mode requires explicit ollama, groq, or gemini provider selection"
            )
        if self.provider is JudgeProvider.OLLAMA:
            self._validate_loopback_endpoint(self.endpoint or _OLLAMA_DEFAULT_ENDPOINT)
        elif self.endpoint is not None:
            raise ValueError(
                "Groq and Gemini endpoints are fixed; arbitrary URLs are not permitted"
            )
        return self

    @property
    def resolved_endpoint(self) -> str | None:
        if self.provider is JudgeProvider.OLLAMA:
            return self.endpoint or _OLLAMA_DEFAULT_ENDPOINT
        if self.provider is JudgeProvider.GROQ:
            return _GROQ_ENDPOINT
        if self.provider is JudgeProvider.GEMINI:
            return _GEMINI_ENDPOINT
        return None

    @property
    def required_key_environment_variable(self) -> str | None:
        if self.provider is JudgeProvider.GROQ:
            return "GROQ_API_KEY"
        if self.provider is JudgeProvider.GEMINI:
            return "GEMINI_API_KEY"
        return None

    def api_key_from(self, environ: dict[str, str] | None = None) -> str | None:
        """Resolve only the explicit provider's key; no key or provider fallback exists."""
        key_name = self.required_key_environment_variable
        if key_name is None:
            return None
        key = (environ or os.environ).get(key_name)
        if not key:
            raise ValueError(f"manual {self.provider.value} judge requires {key_name}")
        return key

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> JudgeSettings:
        values = environ or os.environ
        provider = values.get("JUDGE_PROVIDER", JudgeProvider.RECORDED.value).casefold()
        if provider in _FORBIDDEN_PROVIDER_NAMES:
            raise ValueError("OpenAI, Anthropic, and Azure judge providers are not permitted")
        settings = cls(
            mode=values.get("JUDGE_MODE", JudgeMode.DISABLED.value).casefold(),
            provider=provider,
            endpoint=values.get("JUDGE_OLLAMA_ENDPOINT"),
            model=values.get("JUDGE_MODEL"),
        )
        if settings.mode is JudgeMode.MANUAL:
            settings.api_key_from(values)
        return settings

    @staticmethod
    def _validate_loopback_endpoint(endpoint: str) -> None:
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama endpoint must be an HTTP loopback URL without credentials")
