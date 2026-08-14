"""Stable player-policy observation encoding."""

from __future__ import annotations

import numpy as np

from .constants import (
    ARENA_CENTER,
    ARENA_RADIUS,
    ACTION_COUNT,
    ENEMY_DECISION_SECONDS,
    OBSERVATION_SIZE,
    PLAYER_SPEED,
    ROUND_DURATION_SECONDS,
    SURGE_DURATION_SECONDS,
)


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
