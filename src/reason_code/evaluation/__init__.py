"""Offline deterministic fixture evaluation for explanation regression checks."""

from .runner import EvaluationError, run_deterministic_evaluation

__all__ = ["EvaluationError", "run_deterministic_evaluation"]
