"""Reproducible, non-learning player policies and held-out evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

import numpy as np

from .rules import (
    ARENA_RADIUS,
    PLAYER_DECISION_SECONDS,
    PLAYER_SPEED,
    ROUND_DURATION_SECONDS,
    Direction,
    DIRECTION_UNITS,
)
from .environment import OrbitChasePlayerEnv
from .observation import ObservationView, decode


HELD_OUT_SEED_START = 10_000
DEFAULT_EVALUATION_EPISODES = 100
PELLET_SAFETY_WEIGHT = 0.15


class PlayerPolicy(Protocol):
    """Select one masked player action from a Gym observation."""

    name: str

    def choose(self, state: dict[str, np.ndarray], rng: np.random.Generator) -> int:
        """Return one action allowed by the state action mask."""


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate held-out episode outcomes for one fixed policy."""

    policy: str
    episodes: int
    clears: int
    captures: int
    timeouts: int
    mean_pellets_collected: float
    mean_return: float
    mean_time_to_clear_seconds: float | None

    @property
    def clear_rate(self) -> float:
        return self.clears / self.episodes

    @property
    def capture_rate(self) -> float:
        return self.captures / self.episodes

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.episodes

    def as_dict(self) -> dict:
        """Return metrics in a stable, JSON-ready form."""
        result = asdict(self)
        result.update(
            clear_rate=self.clear_rate,
            capture_rate=self.capture_rate,
            timeout_rate=self.timeout_rate,
        )
        return result


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


POLICIES: dict[str, type[Policy]] = {
    RandomValidPolicy.name: RandomValidPolicy,
    PelletSeekingPolicy.name: PelletSeekingPolicy,
}


def evaluate_policy(policy: PlayerPolicy, seeds: Sequence[int]) -> EvaluationResult:
    """Evaluate one fixed policy on explicit held-out episode seeds."""
    if not seeds:
        raise ValueError("Evaluation requires at least one held-out seed.")

    returns: list[float] = []
    pellets_collected: list[int] = []
    clear_times: list[float] = []
    clears = captures = timeouts = 0

    for seed in seeds:
        episode_return, episode_pellets, info, state = _play_episode(policy, seed)
        returns.append(episode_return)
        pellets_collected.append(episode_pellets)
        if info["captured"]:
            captures += 1
        elif info["cleared"]:
            clears += 1
            clear_times.append(
                (1.0 - decode(state["observation"]).time_fraction) * ROUND_DURATION_SECONDS
            )
        elif info["timed_out"]:
            timeouts += 1

    return EvaluationResult(
        policy=policy.name,
        episodes=len(seeds),
        clears=clears,
        captures=captures,
        timeouts=timeouts,
        mean_pellets_collected=round(float(np.mean(pellets_collected)), 6),
        mean_return=round(float(np.mean(returns)), 6),
        mean_time_to_clear_seconds=(round(float(np.mean(clear_times)), 6) if clear_times else None),
    )


def _play_episode(
    policy: PlayerPolicy, seed: int
) -> tuple[float, int, dict[str, bool | int], dict[str, np.ndarray]]:
    """Run one seeded episode and return its return, pellets, terminal info, and last state."""
    environment = OrbitChasePlayerEnv()
    state, _ = environment.reset(seed=seed)
    rng = np.random.default_rng(seed)
    episode_return = 0.0
    episode_pellets = 0

    while True:
        action = policy.choose(state, rng)
        state, reward, terminated, truncated, info = environment.step(action)
        episode_return += reward
        episode_pellets += info["pellets_collected"]
        if terminated or truncated:
            return episode_return, episode_pellets, info, state


def held_out_seeds(
    episodes: int = DEFAULT_EVALUATION_EPISODES,
    start: int = HELD_OUT_SEED_START,
) -> tuple[int, ...]:
    """Return the fixed seed range reserved for baseline evaluation."""
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive.")
    return tuple(range(start, start + episodes))


def _valid_actions(state: dict[str, np.ndarray]) -> np.ndarray:
    valid_actions = np.flatnonzero(state["action_mask"])
    if len(valid_actions) == 0:
        raise RuntimeError("Environment returned an action mask with no valid actions.")
    return valid_actions


def _action_score(view: ObservationView, action: int, safety_weight: float) -> float:
    unit = DIRECTION_UNITS[Direction(int(action))]
    action_distance = PLAYER_SPEED * PLAYER_DECISION_SECONDS / ARENA_RADIUS
    next_player = view.player + np.asarray(unit) * action_distance
    nearest_pellet_distance = float(np.min(np.linalg.norm(view.pellets - next_player, axis=1)))
    enemy_distance = float(np.linalg.norm(view.enemy - next_player))
    return nearest_pellet_distance - safety_weight * enemy_distance
