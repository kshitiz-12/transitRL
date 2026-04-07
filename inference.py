"""
Run a short episode with a simple greedy policy and print the grader score.
Works offline (direct env API, no HTTP).
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

from env import TransitEnv
from grader import evaluate
from tasks import TASK_MEDIUM

# ------------------ ENV VARIABLES ------------------
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN")  # no default

# ------------------ OPENAI CLIENT ------------------
from openai import OpenAI

client = None
if HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

# ------------------ LOGGING ------------------
def log_step(step, action, reward):
    print(f"STEP {step} | action={action} | reward={reward}")

# ------------------ LOGIC ------------------
def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def greedy_valid_action(env: TransitEnv, rng: random.Random) -> Tuple[int, int]:
    candidates: List[Tuple[int, int, int]] = []
    for d in env.drivers:
        if d.busy:
            continue
        for r in env.riders:
            dist = _manhattan(d.x, d.y, r.x, r.y)
            if dist <= 6:
                candidates.append((d.id, r.id, dist))

    if candidates:
        candidates.sort(key=lambda t: t[2])
        best_d = candidates[0][2]
        best = [c for c in candidates if c[2] == best_d]
        pick = rng.choice(best)
        return pick[0], pick[1]

    if env.drivers and env.riders:
        return rng.choice(env.drivers).id, rng.choice(env.riders).id

    return 0, 0


# ------------------ MAIN ------------------
def main() -> None:
    print("START")

    rng = random.Random()

    env = TransitEnv(task_config=TASK_MEDIUM)
    env.reset()

    done = False
    max_steps = 45
    step_count = 0

    for _ in range(max_steps):
        if done or not env.riders:
            break

        if rng.random() < 0.20:
            d = rng.choice([x for x in env.drivers if not x.busy] or env.drivers)
            r = rng.choice(env.riders)
            action = (d.id, r.id)
        else:
            action = greedy_valid_action(env, rng)

        _, reward, done = env.step(action)

        log_step(step_count, action, reward)
        step_count += 1

    score = evaluate(env)

    print(f"END | score={score:.4f}")


# ------------------
if __name__ == "__main__":
    main()