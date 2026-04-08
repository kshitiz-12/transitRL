"""Grader for the hard task (peak-hour surge)."""

from __future__ import annotations

from env import TransitEnv
from grader import evaluate as _evaluate


def evaluate(env: TransitEnv) -> float:
    return float(_evaluate(env))
