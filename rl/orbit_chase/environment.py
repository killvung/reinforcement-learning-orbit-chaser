"""Thin Gymnasium adapter around the deterministic game simulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .rules import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    PLAYER_DECISION_SECONDS,
    PHYSICS_DT_SECONDS,
    REWARD_CAPTURE,
    REWARD_CLEAR,
    REWARD_PER_DECISION,
    REWARD_PER_ORB,
    REWARD_PER_PELLET,
    REWARD_TIMEOUT,
    Direction,
)
from .observation import encode
from .simulation import GameSimulation, StepEvents


class OrbitChasePlayerEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium player agent; the red enemy remains deterministic."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32),
            "action_mask": spaces.MultiBinary(ACTION_COUNT),
        })
        self.simulation = GameSimulation()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Start a deterministic episode and return its encoded policy state."""
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**32)) if seed is None else seed
        self.simulation.reset(episode_seed)
        return self._observation(), {"seed": episode_seed}

    def step(self, action: int):
        """Hold one player direction for the configured 100 ms decision interval."""
        direction = self._direction_from_action(action)
        events = self._run_decision_interval(direction)
        terminated = events.captured or events.cleared
        truncated = events.timed_out and not terminated
        return self._observation(), self._reward(events), terminated, truncated, self._info(events)

    def _run_decision_interval(self, direction: Direction) -> StepEvents:
        """Advance ten ticks and aggregate pickup events across the interval."""
        pellets_collected = 0
        orbs_collected = 0
        final_events = StepEvents()
        ticks_per_decision = round(PLAYER_DECISION_SECONDS / PHYSICS_DT_SECONDS)
        for _ in range(ticks_per_decision):
            final_events = self.simulation.step(direction)
            pellets_collected += final_events.pellets_collected
            orbs_collected += final_events.orbs_collected
            if final_events.captured or final_events.cleared or final_events.timed_out:
                break
        return StepEvents(
            pellets_collected=pellets_collected,
            orbs_collected=orbs_collected,
            captured=final_events.captured,
            cleared=final_events.cleared,
            timed_out=final_events.timed_out,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "observation": encode(self.simulation),
            "action_mask": np.asarray(self.simulation.valid_actions(), dtype=np.int8),
        }

    def _direction_from_action(self, action: int) -> Direction:
        """Validate Gym's integer action before converting it to a direction."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        return Direction(action)

    @staticmethod
    def _reward(events: StepEvents) -> float:
        """Map game facts to the player learning objective."""
        reward = REWARD_PER_DECISION
        reward += events.pellets_collected * REWARD_PER_PELLET
        reward += events.orbs_collected * REWARD_PER_ORB
        # Match browser outcome precedence: contact with the enemy is a loss,
        # even if the final pellet is collected on that same physics tick.
        if events.captured:
            reward += REWARD_CAPTURE
        elif events.cleared:
            reward += REWARD_CLEAR
        elif events.timed_out:
            reward += REWARD_TIMEOUT
        return reward

    @staticmethod
    def _info(events: StepEvents) -> dict[str, bool | int]:
        """Expose factual episode events for debugging and evaluation."""
        return {
            "captured": events.captured,
            "cleared": events.cleared,
            "timed_out": events.timed_out,
            "pellets_collected": events.pellets_collected,
            "orbs_collected": events.orbs_collected,
        }
