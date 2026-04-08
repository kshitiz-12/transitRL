"""
Curriculum tasks: control supply/demand and dynamics for TransitRL.

EASY: low rider demand, many drivers (easy matching).
MEDIUM: balanced counts.
HARD: peak-hour surge — many riders, few drivers, slower ride completions.
"""

from typing import Any, Dict

# Low demand, high supply
TASK_EASY: Dict[str, Any] = {
    "drivers_range": (8, 10),
    "riders_range": (3, 5),
    "completion_prob": 0.45,
    "max_steps": 200,
}

# Balanced
TASK_MEDIUM: Dict[str, Any] = {
    "drivers_range": (5, 8),
    "riders_range": (5, 8),
    "completion_prob": 0.3,
    "max_steps": 200,
}

# High demand, limited drivers, harder to clear backlog
TASK_HARD: Dict[str, Any] = {
    "drivers_range": (3, 5),
    "riders_range": (8, 10),
    "completion_prob": 0.18,
    "max_steps": 250,
}

TASKS = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "easy": "Low demand and high driver availability.",
    "medium": "Balanced rider demand and driver supply.",
    "hard": "Peak-hour surge with high demand and limited drivers.",
}


def get_task(name: str) -> Dict[str, Any]:
    """Return task config by name; defaults to medium if unknown."""
    return TASKS.get(name.lower(), TASK_MEDIUM)


def list_tasks_with_graders() -> list[Dict[str, Any]]:
    """
    Expose at least three tasks with explicit grader wiring for validators.
    """
    result: list[Dict[str, Any]] = []
    for task_id in ("easy", "medium", "hard"):
        result.append(
            {
                "id": task_id,
                "name": f"TransitRL {task_id.title()}",
                "description": TASK_DESCRIPTIONS[task_id],
                "difficulty": task_id,
                "grader": "grader:evaluate",
                "enabled": True,
            }
        )
    return result
