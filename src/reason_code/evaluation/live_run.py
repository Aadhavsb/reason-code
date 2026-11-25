"""Manual CLI for the isolated one-request live LLM experiment."""

from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT_OUTPUT_TOKEN_CAP, DEFAULT_TIMEOUT_SECONDS, JudgeSettings
from .live_runner import LIVE_REQUEST_CAP, run_live_experiment


def _json_line(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _configuration_failure() -> dict[str, object]:
    return {
        "metadata": {"n": 1, "non_authoritative": True, "request_cap": LIVE_REQUEST_CAP},
        "rejection_cause": "configuration",
        "status": "fail",
        "verdict": {
            "accepted": False,
            "faithful": False,
            "findings": ["Live experiment configuration was rejected; no request was made."],
        },
    }


def main() -> int:
    """Require an affirmative local command before a single provider request."""
    parser = argparse.ArgumentParser(
        description="Run one manual LLM request against one named synthetic fixture."
    )
    parser.add_argument("--confirm-live-request", action="store_true")
    parser.add_argument("--provider", choices=("ollama", "groq", "gemini"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-endpoint")
    parser.add_argument("--output-token-cap", type=int, default=DEFAULT_OUTPUT_TOKEN_CAP)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if not args.confirm_live_request:
        print(_json_line(_configuration_failure()))
        print("live experiment rejected: --confirm-live-request is required", file=sys.stderr)
        return 2

    try:
        settings = JudgeSettings(
            mode="manual",
            provider=args.provider,
            model=args.model,
            endpoint=args.ollama_endpoint,
            output_token_cap=args.output_token_cap,
            timeout_seconds=args.timeout_seconds,
        )
        report = run_live_experiment(settings)
    except Exception:
        # Never serialize exception messages: they may include provider/transport details.
        print(_json_line(_configuration_failure()))
        print("live experiment rejected: invalid configuration", file=sys.stderr)
        return 1

    print(_json_line(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
