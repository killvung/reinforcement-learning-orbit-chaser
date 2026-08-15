"""Shared learning-agent contract for linear and neural Sarsa."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .rules import ACTION_COUNT


@dataclass(frozen=True)
class Transition:
    """One Gym decision: (s, a, r, s', a') with a terminal flag.

    One-step Sarsa uses `next_action`. Neural Expected Sarsa ignores it and
    bootstraps from the current policy's expected value at `next_state`.
    """

    state: dict[str, np.ndarray]
    action: int
    reward: float
    next_state: dict[str, np.ndarray] | None
    next_action: int | None
    terminal: bool


class Agent(ABC):
    """On-policy action-value learner with a shared masked epsilon-greedy policy.

    Subclasses estimate unmasked `Q(s, ·)` in `Direction` order. This base
    class applies the Gym action mask only during selection: greedy choice
    looks at valid actions only, and exploration samples uniformly among them.
    A later neural Expected-Sarsa agent implements the same `q_values` /
    `update` / `reset_episode` surface.
    """

    name: str

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def q_values(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Return unmasked Q(s, a) for a in Direction order, shape (8,)."""

    @abstractmethod
    def update(self, transition: Transition) -> float | None:
        """Learn from one decision. Return the TD error when the algorithm has one."""

    def reset_episode(self) -> None:
        """Clear episode-local learning state (traces, n-step buffer, q_old)."""

    def snapshot(self) -> dict:
        """Return inspectable scalars for training logs."""
        return {}

    def select_action(self, state: dict[str, np.ndarray], epsilon: float) -> int:
        """Masked epsilon-greedy. Lowest valid index wins greedy ties.

        With probability epsilon, sample uniformly from valid actions. Otherwise
        pick argmax Q(s, a) among valid a. This is the behaviour policy for
        on-policy Sarsa.
        """
        return self._select_from_q(self.q_values(state), state["action_mask"], epsilon)

    def choose(self, state: dict[str, np.ndarray], _rng: np.random.Generator) -> int:
        """Greedy masked action for held-out evaluation (`PlayerPolicy`)."""
        return self.select_action(state, epsilon=0.0)

    def _select_from_q(
        self, values: np.ndarray, action_mask: np.ndarray, epsilon: float
    ) -> int:
        mask = np.asarray(action_mask, dtype=np.int8)
        if mask.shape != (ACTION_COUNT,):
            raise ValueError(f"Expected action mask shape ({ACTION_COUNT},), got {mask.shape}.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Action mask values must be binary.")
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise ValueError("Action mask must contain at least one valid action.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must lie in [0, 1].")
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        best_value = np.max(values[valid])
        return int(valid[np.flatnonzero(values[valid] == best_value)[0]])
