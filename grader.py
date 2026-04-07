"""
Deterministic episode score in [0.0, 1.0] from environment metrics.

Weighted formula:
score = 0.5 * service_ratio + 0.3 * (1 - normalized_avg_wait_time) + 0.2 * normalized_throughput
"""

from __future__ import annotations

from env import TransitEnv


def evaluate(env: TransitEnv) -> float:
    """
    Score reflects completed rides, average wait at match, and throughput.
    Uses deterministic episode counters:
    served_count, total_requests, avg_wait_time, step_count.
    """
    served_count = max(0, int(env.served_count))
    total_requests = max(1, int(getattr(env, "total_requests", env.initial_rider_count)))
    step_count = max(1, int(env.step_count))

    avg_wait_time = (
        float(env.total_wait_at_serve) / served_count if served_count > 0 else float(step_count)
    )

    service_ratio = min(1.0, served_count / total_requests)
    normalized_avg_wait_time = min(1.0, avg_wait_time / 20.0)
    normalized_throughput = min(1.0, served_count / step_count)

    score = (
        0.5 * service_ratio
        + 0.3 * (1.0 - normalized_avg_wait_time)
        + 0.2 * normalized_throughput
    )
    return float(max(0.0, min(1.0, score)))
