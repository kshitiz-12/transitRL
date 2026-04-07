"""Pydantic request/response models for the TransitRL HTTP API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    driver_id: int = Field(..., description="Assigned driver id")
    rider_id: int = Field(..., description="Rider id to serve")


class StateResponse(BaseModel):
    drivers: List[Dict[str, Any]]
    riders: List[Dict[str, Any]]
    time: int


class StepResponse(BaseModel):
    next_state: StateResponse
    reward: float
    done: bool


class ResetRequest(BaseModel):
    task: str = Field(default="medium", description="Task key: easy, medium, hard")
    seed: Optional[int] = Field(default=None, description="RNG seed for reproducibility")


class ResetResponse(BaseModel):
    state: StateResponse
