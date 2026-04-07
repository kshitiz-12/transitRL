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


def get_task(name: str) -> Dict[str, Any]:
    """Return task config by name; defaults to medium if unknown."""
    return TASKS.get(name.lower(), TASK_MEDIUM)
