"""Stable player-policy observation encoding and decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import (
    ARENA_CENTER,
    ARENA_RADIUS,
    ACTION_COUNT,
    ENEMY_DECISION_SECONDS,
    OBSERVATION_SIZE,
    PELLET_COUNT,
    PLAYER_SPEED,
    ROUND_DURATION_SECONDS,
    SURGE_DURATION_SECONDS,
)

PLAYER_FEATURE_COUNT = 5
ENEMY_FEATURE_COUNT = 11
BAR_FEATURE_COUNT = 8
COLLECTIBLE_FEATURE_WIDTH = 3
TIME_FEATURE_COUNT = 1

PLAYER_POSITION_SLICE = slice(0, 2)
ENEMY_POSITION_SLICE = slice(PLAYER_FEATURE_COUNT, PLAYER_FEATURE_COUNT + 2)
PELLET_FEATURE_OFFSET = PLAYER_FEATURE_COUNT + ENEMY_FEATURE_COUNT + BAR_FEATURE_COUNT


@dataclass(frozen=True)
class ObservationView:
    """Named slices of the encoded player observation vector."""

    player: np.ndarray
    enemy: np.ndarray
    pellets: np.ndarray
    time_fraction: float


def encode(simulation) -> np.ndarray:
    """Encode simulation state into the documented, ordered 130-value vector."""
    values = [
        *_player_features(simulation),
        *_enemy_features(simulation),
        *_bar_features(simulation),
        *_collectible_features(simulation.arena.pellet_slots, simulation.pellet_active),
        *_collectible_features(simulation.arena.orb_slots, simulation.orb_active),
        _time_feature(simulation),
    ]
    if len(values) != OBSERVATION_SIZE:
        raise ValueError(
            f"Expected {OBSERVATION_SIZE} observation values, got {len(values)}."
        )
    return np.asarray(values, dtype=np.float32)


def decode(observation: np.ndarray) -> ObservationView:
    """Read player, enemy, active pellets, and remaining-time fraction."""
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"Expected {OBSERVATION_SIZE} observation values, got {observation.shape}."
        )
    pellet_features = observation[
        PELLET_FEATURE_OFFSET : PELLET_FEATURE_OFFSET + PELLET_COUNT * COLLECTIBLE_FEATURE_WIDTH
    ].reshape(PELLET_COUNT, COLLECTIBLE_FEATURE_WIDTH)
    return ObservationView(
        player=observation[PLAYER_POSITION_SLICE],
        enemy=observation[ENEMY_POSITION_SLICE],
        pellets=pellet_features[pellet_features[:, 2] > 0.5, :2],
        time_fraction=float(observation[-TIME_FEATURE_COUNT]),
    )


def _player_features(simulation) -> list[float]:
    """Encode position, velocity, and remaining Surge time (indices 0–4)."""
    return [
        *_normalize_point((simulation.player.x, simulation.player.y)),
        simulation.player_velocity[0] / PLAYER_SPEED,
        simulation.player_velocity[1] / PLAYER_SPEED,
        simulation.surge_remaining / SURGE_DURATION_SECONDS,
    ]


def _enemy_features(simulation) -> list[float]:
    """Encode position, heading, and next-decision clock (indices 5–15)."""
    heading = [float(index == simulation.enemy_action) for index in range(ACTION_COUNT)]
    return [
        *_normalize_point((simulation.enemy.x, simulation.enemy.y)),
        *heading,
        max(0.0, simulation.enemy_remaining) / ENEMY_DECISION_SECONDS,
    ]


def _bar_features(simulation) -> list[float]:
    """Encode both obstacle-bar endpoints (indices 16–23)."""
    return [
        coordinate
        for start, end in simulation.arena.bars
        for coordinate in (*_normalize_point(start), *_normalize_point(end))
    ]


def _collectible_features(slots, active_flags) -> list[float]:
    """Encode fixed collectible positions and their active flags."""
    return [
        coordinate
        for slot, active in zip(slots, active_flags)
        for coordinate in (*_normalize_point(slot.position), float(active))
    ]


def _time_feature(simulation) -> float:
    """Encode the non-negative remaining-round fraction (index 129)."""
    return max(0.0, simulation.time_remaining) / ROUND_DURATION_SECONDS


def _normalize_point(point) -> tuple[float, float]:
    """Convert a world-space point to the arena-centred policy coordinate frame."""
    return (
        (point[0] - ARENA_CENTER[0]) / ARENA_RADIUS,
        (point[1] - ARENA_CENTER[1]) / ARENA_RADIUS,
    )
