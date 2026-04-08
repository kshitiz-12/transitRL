import os
import random

# ---------------- SAFE OPENAI IMPORT ----------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------- ENV IMPORTS ----------------
from env import TransitEnv
from grader import evaluate
from tasks import TASK_MEDIUM

# ---------------- ENV VARIABLES ----------------
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# optional client (safe)
if OpenAI and HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
else:
    client = None


# ---------------- HELPER ----------------
def manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def greedy_action(env, rng):
    candidates = []

    for d in env.drivers:
        if d.busy:
            continue
        for r in env.riders:
            dist = manhattan(d.x, d.y, r.x, r.y)
            if dist <= 6:
                candidates.append((d.id, r.id, dist))

    if candidates:
        candidates.sort(key=lambda x: x[2])
        best_dist = candidates[0][2]
        best = [c for c in candidates if c[2] == best_dist]
        d_id, r_id, _ = rng.choice(best)
        return d_id, r_id

    if env.drivers and env.riders:
        return rng.choice(env.drivers).id, rng.choice(env.riders).id

    return 0, 0


# ---------------- MAIN ----------------
def main():
    rng = random.Random()

    env = TransitEnv(task_config=TASK_MEDIUM)
    env.reset()

    done = False
    max_steps = 45
    step_count = 0

    # 🔥 STRICT FORMAT START
    print("[START] task=transitrl", flush=True)

    try:
        for _ in range(max_steps):
            if done or not env.riders:
                break

            if rng.random() < 0.2:
                available = [x for x in env.drivers if not x.busy] or env.drivers
                d = rng.choice(available)
                r = rng.choice(env.riders)
                action = (d.id, r.id)
            else:
                action = greedy_action(env, rng)

            _, reward, done = env.step(action)

            step_count += 1

            # 🔥 STRICT STEP FORMAT
            print(f"[STEP] step={step_count} reward={reward:.4f}", flush=True)

        score = evaluate(env)

        # 🔥 STRICT END FORMAT
        print(f"[END] task=transitrl score={score:.4f} steps={step_count}", flush=True)

    except Exception:
        # 💀 NEVER CRASH
        print(f"[END] task=transitrl score=0.0000 steps={step_count}", flush=True)


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()