"""
FastAPI server exposing TransitRL as an HTTP API (OpenEnv-friendly).
"""

from __future__ import annotations

from typing import Optional

from fastapi import Body, FastAPI, HTTPException

from env import TransitEnv
from models import ResetRequest, ResetResponse, StateResponse
from tasks import get_task, list_tasks_with_graders

app = FastAPI(title="TransitRL", version="1.0.0")

_env: Optional[TransitEnv] = None


def _ensure_env() -> TransitEnv:
    global _env
    if _env is None:
        _env = TransitEnv(task_config=get_task("medium"), seed=0)
        _env.reset(seed=0)
    return _env


@app.post("/reset", response_model=ResetResponse, status_code=200)
def reset(body: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    """Initialize or re-initialize the environment and return the starting state."""
    global _env
    req = body
    seed = req.seed if req.seed is not None else 0
    _env = TransitEnv(task_config=get_task(req.task), seed=seed)
    state = _env.reset(seed=seed)
    return ResetResponse(state=StateResponse(**state))


@app.post("/step", status_code=200)
def step(action: dict = Body(...)) -> dict:
    """Apply one assignment action; response keys are state, reward, done."""
    env = _ensure_env()
    if set(action.keys()) != {"driver_id", "rider_id"}:
        raise HTTPException(
            status_code=422, detail="Action must contain only driver_id and rider_id"
        )
    if not isinstance(action["driver_id"], int) or not isinstance(action["rider_id"], int):
        raise HTTPException(status_code=422, detail="driver_id and rider_id must be int")

    state, reward, done = env.step(action)
    return {"state": state, "reward": float(reward), "done": bool(done)}


@app.get("/state", response_model=StateResponse, status_code=200)
def get_state() -> StateResponse:
    """Return the current environment state (requires a prior reset)."""
    env = _ensure_env()
    state = env.get_state()
    return StateResponse(**state)


@app.get("/tasks", status_code=200)
def get_tasks() -> dict:
    """
    Task catalog for OpenEnv validators.
    """
    return {"tasks": list_tasks_with_graders()}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
