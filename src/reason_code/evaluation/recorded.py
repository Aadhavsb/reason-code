"""Recorded-only explanation judge used by deterministic, no-network evaluation."""

from __future__ import annotations

from collections.abc import Iterable

from .models import JudgeRequest, JudgeResponse


class RecordedResponseJudge:
    """Serve frozen synthetic responses only when all request bindings match exactly.

    A case match with a changed digest or rubric is deliberately not treated as a
    near-match. The returned rejection is fail-closed and cannot be interpreted as a
    passing evaluation.
    """

    def __init__(self, responses: Iterable[JudgeResponse]) -> None:
        frozen_responses = tuple(responses)
        if any(
            response.source != "recorded" or not response.synthetic for response in frozen_responses
        ):
            raise ValueError("recorded judge responses must be synthetic recorded responses")
        self._responses = {response.case_id: response for response in frozen_responses}
        if len(self._responses) != len(frozen_responses):
            raise ValueError("recorded judge responses must have unique case_id values")

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        response = self._responses.get(request.case_id)
        if response is None:
            return self._reject(request, "No recorded response exists for this case_id.")
        if response.evidence_digest != request.evidence_digest:
            return self._reject(
                request, "Recorded response evidence_digest does not match request."
            )
        if response.rubric_version != request.rubric_version:
            return self._reject(request, "Recorded response rubric_version does not match request.")
        return response

    @staticmethod
    def _reject(request: JudgeRequest, finding: str) -> JudgeResponse:
        return JudgeResponse(
            case_id=request.case_id,
            evidence_digest=request.evidence_digest,
            rubric_version=request.rubric_version,
            accepted=False,
            faithful=False,
            findings=(finding,),
            source="recorded",
            synthetic=True,
        )
