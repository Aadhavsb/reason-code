"""Stable evidence digests used to bind recorded judge results."""

from __future__ import annotations

import hashlib
import json

from reason_code.explanations import ExplanationEvidence


def evidence_digest(evidence: ExplanationEvidence) -> str:
    """Return the SHA-256 digest of canonical, validated explanation evidence."""
    payload = json.dumps(
        evidence.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
