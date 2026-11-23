"""Command-line entry point for the offline deterministic evaluation gate."""

from __future__ import annotations

import argparse
import json

from .runner import EvaluationError, run_deterministic_evaluation


def _json_line(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def main() -> int:
    """Run the only supported evaluation mode without acquiring external data."""
    parser = argparse.ArgumentParser(description="Run frozen deterministic explanation fixtures.")
    parser.add_argument("--mode", choices=("deterministic",), required=True)
    parser.parse_args()
    try:
        report = run_deterministic_evaluation()
    except EvaluationError:
        print(_json_line({"mode": "deterministic", "status": "fail"}))
        return 1
    print(_json_line(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
