"""Grader for the medium task (balanced demand/supply)."""

from __future__ import annotations

from env import TransitEnv
from grader import evaluate as _evaluate


def evaluate(env: TransitEnv) -> float:
    return float(_evaluate(env))
