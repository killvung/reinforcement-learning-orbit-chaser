"""Non-learning player policies that select one masked Gym action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .rules import (
    ARENA_RADIUS,
    DIRECTION_UNITS,
    PLAYER_DECISION_SECONDS,
    PLAYER_SPEED,
    Direction,
)
from .observation import ObservationView, decode


PELLET_SAFETY_WEIGHT = 0.15


class PlayerPolicy(Protocol):
    """Select one masked player action from a Gym observation."""

    name: str

    def choose(self, state: dict[str, np.ndarray], rng: np.random.Generator) -> int:
        """Return one action allowed by the state action mask."""


class Policy(ABC):
    """Non-learning selector that may only return a currently valid action."""

    name: str

    def choose(self, state: dict[str, np.ndarray], rng: np.random.Generator) -> int:
        valid = _valid_actions(state)
        action = int(self.select(state, valid, rng))
        mask = state["action_mask"]
        if action < 0 or action >= len(mask) or not mask[action]:
            raise RuntimeError(f"{self.name} selected masked action {action}.")
        return action

    @abstractmethod
    def select(
        self,
        state: dict[str, np.ndarray],
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Return one index from `valid`."""


class RandomValidPolicy(Policy):
    """Uniformly sample one currently valid action."""

    name = "random-valid"

    def select(
        self,
        _state: dict[str, np.ndarray],
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        return int(rng.choice(valid))


class PelletSeekingPolicy(Policy):
    """Move toward the nearest active pellet while retaining enemy distance."""

    name = "pellet-seeking"

    def __init__(self, safety_weight: float = PELLET_SAFETY_WEIGHT) -> None:
        self.safety_weight = safety_weight

    def select(
        self,
        state: dict[str, np.ndarray],
        valid: np.ndarray,
        _rng: np.random.Generator,
    ) -> int:
        view = decode(state["observation"])
        if len(view.pellets) == 0:
            return int(valid[0])
        return int(min(valid, key=lambda action: _action_score(view, action, self.safety_weight)))


class EnemyEvadePolicy(Policy):
    """Move to maximize distance from the enemy after the next 100 ms step."""

    name = "enemy-evade"

    def select(
        self,
        state: dict[str, np.ndarray],
        valid: np.ndarray,
        _rng: np.random.Generator,
    ) -> int:
        view = decode(state["observation"])
        return int(min(valid, key=lambda action: _evade_score(view, action)))


POLICIES: dict[str, type[Policy]] = {
    RandomValidPolicy.name: RandomValidPolicy,
    PelletSeekingPolicy.name: PelletSeekingPolicy,
    EnemyEvadePolicy.name: EnemyEvadePolicy,
}


def _valid_actions(state: dict[str, np.ndarray]) -> np.ndarray:
    """Return the currently unmasked action indices, or fail if none exist."""
    valid_actions = np.flatnonzero(state["action_mask"])
    if len(valid_actions) == 0:
        raise RuntimeError("Environment returned an action mask with no valid actions.")
    return valid_actions


def _projected_player(view: ObservationView, action: int) -> np.ndarray:
    """Estimate the player's position after one 100 ms move in `action`."""
    dx, dy = DIRECTION_UNITS[Direction(int(action))]
    step = PLAYER_SPEED * PLAYER_DECISION_SECONDS / ARENA_RADIUS
    return view.player + np.asarray((dx, dy)) * step


def _action_score(view: ObservationView, action: int, safety_weight: float) -> float:
    """Prefer nearer pellets, with a penalty for closing on the enemy."""
    next_player = _projected_player(view, action)
    nearest_pellet_distance = float(np.min(np.linalg.norm(view.pellets - next_player, axis=1)))
    enemy_distance = float(np.linalg.norm(view.enemy - next_player))
    return nearest_pellet_distance - safety_weight * enemy_distance


def _evade_score(view: ObservationView, action: int) -> float:
    """Return the negated enemy distance so minimizing this score flees."""
    return -float(np.linalg.norm(view.enemy - _projected_player(view, action)))
