"""
TransitRL environment: grid-based ride-hailing assignment simulation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Grid inclusive bounds [0, GRID_MAX]
GRID_MAX = 10

# Distance constraints
SOFT_RADIUS = 3
HARD_MAX_RADIUS = 6
MAX_RIDERS = 20


@dataclass
class Driver:
    id: int
    x: int
    y: int
    busy: bool = False
    idle_time: int = 0
    rides_completed: int = 0


@dataclass
class Rider:
    id: int
    x: int
    y: int
    wait_time: int = 0


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def _default_task() -> Dict[str, Any]:
    return {
        "drivers_range": (5, 10),
        "riders_range": (5, 10),
        "completion_prob": 0.3,
    }


class TransitEnv:
    """
    OpenEnv-style RL environment: assign drivers to riders on a 2D grid.
    """

    def __init__(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.task_config = task_config or _default_task()
        self.time: int = 0
        self.drivers: List[Driver] = []
        self.riders: List[Rider] = []
        self._completion_prob: float = float(self.task_config.get("completion_prob", 0.3))

        # Grader metrics (updated during episode)
        self.initial_rider_count: int = 0
        self.total_requests: int = 0
        self.served_count: int = 0
        self.total_wait_at_serve: int = 0
        self.step_count: int = 0
        self.invalid_action_count: int = 0
        self.rejected_radius_count: int = 0
        self._next_rider_id: int = 0
        self._next_spawn_in: int = 2

    def _sample_int(self, key: str, default: Tuple[int, int]) -> int:
        lo, hi = self.task_config.get(key, default)
        return self._rng.randint(int(lo), int(hi))

    def _place_entity(self) -> Tuple[int, int]:
        return self._rng.randint(0, GRID_MAX), self._rng.randint(0, GRID_MAX)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            # Keep global and local RNG aligned for reproducibility.
            random.seed(seed)
            self._rng = random.Random(seed)

        n_drivers = self._sample_int("drivers_range", (5, 10))
        n_riders = self._sample_int("riders_range", (5, 10))

        self.drivers = []
        for i in range(n_drivers):
            x, y = self._place_entity()
            self.drivers.append(Driver(id=i, x=x, y=y, busy=False, idle_time=0))

        self.riders = []
        for j in range(n_riders):
            x, y = self._place_entity()
            self.riders.append(Rider(id=j, x=x, y=y, wait_time=0))

        self.time = 0
        self.initial_rider_count = len(self.riders)
        self.total_requests = len(self.riders)
        self.served_count = 0
        self.total_wait_at_serve = 0
        self.step_count = 0
        self.invalid_action_count = 0
        self.rejected_radius_count = 0
        self._next_rider_id = len(self.riders)
        self._next_spawn_in = self._rng.randint(2, 3)

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "drivers": [self._driver_to_dict(d) for d in self.drivers],
            "riders": [self._rider_to_dict(r) for r in self.riders],
            "time": self.time,
        }

    @staticmethod
    def _driver_to_dict(d: Driver) -> Dict[str, Any]:
        return {
            "id": d.id,
            "x": d.x,
            "y": d.y,
            "busy": d.busy,
            "idle_time": d.idle_time,
            "rides_completed": d.rides_completed,
        }

    @staticmethod
    def _rider_to_dict(r: Rider) -> Dict[str, Any]:
        return {"id": r.id, "x": r.x, "y": r.y, "wait_time": r.wait_time}

    def _find_driver(self, driver_id: int) -> Optional[Driver]:
        for d in self.drivers:
            if d.id == driver_id:
                return d
        return None

    def _find_rider(self, rider_id: int) -> Optional[Rider]:
        for r in self.riders:
            if r.id == rider_id:
                return r
        return None

    def _parse_action(
        self, action: Union[Tuple[int, int], Dict[str, int], List[int]]
    ) -> Optional[Tuple[int, int]]:
        if isinstance(action, (tuple, list)):
            if len(action) != 2:
                return None
            return int(action[0]), int(action[1])
        if isinstance(action, dict):
            if "driver_id" not in action or "rider_id" not in action:
                return None
            return int(action["driver_id"]), int(action["rider_id"])
        return None

    def step(
        self, action: Union[Tuple[int, int], Dict[str, int], List[int]]
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        One environment step: try (driver_id, rider_id), then advance time.
        Returns (next_state, reward, done).
        """
        parsed = self._parse_action(action)
        reward = 0.0
        demand_pressure = len(self.riders) > len(self.drivers)

        if not self.riders:
            # Episode already terminal when no riders remain.
            return self.get_state(), 0.0, True

        if parsed is None:
            self.invalid_action_count += 1
            reward = -5.0
            if demand_pressure:
                reward -= 0.5
            reward += self._fairness_reward_adjustment()
            self._advance_time()
            return self.get_state(), reward, self._is_done()

        driver_id, rider_id = parsed
        driver = self._find_driver(driver_id)
        rider = self._find_rider(rider_id)

        if driver is None or rider is None:
            self.invalid_action_count += 1
            reward = -5.0
            if demand_pressure:
                reward -= 0.5
            reward += self._fairness_reward_adjustment()
            self._advance_time()
            return self.get_state(), reward, self._is_done()

        if driver.busy:
            self.invalid_action_count += 1
            reward = -5.0
            if demand_pressure:
                reward -= 0.5
            reward += self._fairness_reward_adjustment()
            self._advance_time()
            return self.get_state(), reward, self._is_done()

        dist = _manhattan(driver.x, driver.y, rider.x, rider.y)
        if dist > HARD_MAX_RADIUS:
            self.rejected_radius_count += 1
            reward = -10.0
            if demand_pressure:
                reward -= 0.5
            reward += self._fairness_reward_adjustment()
            self._advance_time()
            return self.get_state(), reward, self._is_done()

        # Driver-side acceptance behavior:
        # farther pickups are less likely to be accepted.
        accept_prob = 1.0
        if dist > 3:
            accept_prob -= 0.2
        if dist > 5:
            accept_prob -= 0.4
        accept_prob = max(0.1, min(1.0, accept_prob))
        if self._rng.random() > accept_prob:
            reward -= 5.0
            reward += self._fairness_reward_adjustment()
            return self.get_state(), reward, False

        # Valid assignment: reward shaped by distance, wait, idle, soft radius
        w = rider.wait_time
        i_idle = driver.idle_time
        reward = 10.0
        reward -= 0.4 * dist
        reward -= 0.25 * w
        reward -= 0.15 * i_idle
        if dist > SOFT_RADIUS:
            reward -= 0.5 * (dist - SOFT_RADIUS)
        # Demand-aware shaping for surge situations.
        if demand_pressure:
            reward -= 0.5
            reward += 1.0

        # Post-accept cancellation behavior:
        # even accepted rides can cancel before pickup completion.
        if self._rng.random() < 0.1:
            reward -= 3.0
            reward += self._fairness_reward_adjustment()
            return self.get_state(), reward, False

        driver.busy = True
        # Fairness tracking: count successful assignments per driver.
        driver.rides_completed += 1
        self.riders = [r for r in self.riders if r.id != rider_id]

        self.served_count += 1
        self.total_wait_at_serve += w
        reward += self._fairness_reward_adjustment()

        self._advance_time()
        return self.get_state(), reward, self._is_done()

    def _is_done(self) -> bool:
        return self.time >= 50 or not self.riders

    def _advance_time(self) -> None:
        self.time += 1
        self.step_count += 1

        for r in self.riders:
            r.wait_time += 1

        for d in self.drivers:
            if d.busy:
                if self._rng.random() < self._completion_prob:
                    d.busy = False
                    d.idle_time = 0
            else:
                d.idle_time += 1
                # Idle drivers reposition locally to simulate search behavior.
                d.x = self._clamp(d.x + self._rng.randint(-1, 1))
                d.y = self._clamp(d.y + self._rng.randint(-1, 1))

        # Dynamic demand: every 2-3 steps, spawn 1-2 riders (bounded by MAX_RIDERS).
        self._next_spawn_in -= 1
        if self._next_spawn_in <= 0:
            self._spawn_riders()
            self._next_spawn_in = self._rng.randint(2, 3)

    @staticmethod
    def _clamp(v: int) -> int:
        return max(0, min(GRID_MAX, v))

    def _spawn_riders(self) -> None:
        available_slots = max(0, MAX_RIDERS - len(self.riders))
        if available_slots == 0:
            return

        spawn_count = min(self._rng.randint(1, 2), available_slots)
        for _ in range(spawn_count):
            x, y = self._place_entity()
            self.riders.append(Rider(id=self._next_rider_id, x=x, y=y, wait_time=0))
            self._next_rider_id += 1
            self.total_requests += 1

    def _fairness_reward_adjustment(self) -> float:
        """
        Fairness in ride-hailing:
        keep ride allocations balanced across drivers so a few drivers do not
        monopolize requests while others stay underutilized.
        This makes reward multi-objective: efficiency + equity.
        """
        if not self.drivers:
            return 0.0
        rides = [d.rides_completed for d in self.drivers]
        imbalance = max(rides) - min(rides)
        if imbalance > 3:
            # Penalize highly uneven distribution.
            return -1.0
        if imbalance <= 1:
            # Small bonus for balanced allocations.
            return 0.5
        return 0.0
