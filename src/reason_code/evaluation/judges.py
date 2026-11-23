"""Local-only judge construction; manual providers are never auto-invoked."""

from __future__ import annotations

from collections.abc import Iterable

from .config import JudgeMode, JudgeSettings
from .models import JudgeRequest, JudgeResponse
from .recorded import RecordedResponseJudge


class DisabledJudge:
    """Fail closed when optional judging has not been explicitly enabled."""

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        return JudgeResponse(
            case_id=request.case_id,
            evidence_digest=request.evidence_digest,
            rubric_version=request.rubric_version,
            accepted=False,
            faithful=False,
            findings=("Optional explanation judge is disabled.",),
            source="disabled",
            synthetic=False,
        )


def build_local_judge(
    settings: JudgeSettings, responses: Iterable[JudgeResponse] = ()
) -> DisabledJudge | RecordedResponseJudge:
    """Build only disabled or recorded judges.

    Manual Ollama, Groq, and Gemini settings are configuration-only. A host must
    deliberately implement and invoke a manual run; this package never falls back to
    a network provider and the normal evaluation path remains network-free.
    """
    if settings.mode is JudgeMode.DISABLED:
        return DisabledJudge()
    if settings.mode is JudgeMode.RECORDED:
        return RecordedResponseJudge(responses)
    raise ValueError("manual judge providers require an explicit host-controlled invocation")
