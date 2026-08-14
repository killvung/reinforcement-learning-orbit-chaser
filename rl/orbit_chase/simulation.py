"""Gym-free, deterministic fixed-tick Orbit Chase game rules."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .arena import Arena, make_arena
from .constants import (
    DIRECTION_VECTORS,
    ENEMY_RADIUS,
    ENEMY_SPEED,
    ENEMY_START,
    PHYSICS_DT_SECONDS,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    PLAYER_START,
    Direction,
)
from .geometry import is_blocked


@dataclass
class Actor:
    """Mutable circular actor state used by the physics simulation."""

    x: float
    y: float
    radius: float
    speed: float


class GameSimulation:
    """Owns deterministic arena state and one fixed physics tick at a time."""

    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int) -> None:
        """Create a fresh deterministic episode from a seed."""
        self.arena: Arena = make_arena(seed)
        self.player = Actor(*PLAYER_START, PLAYER_RADIUS, PLAYER_SPEED)
        self.enemy = Actor(*ENEMY_START, ENEMY_RADIUS, ENEMY_SPEED)
        self.player_velocity = (0.0, 0.0)

    def move_player(self, direction: Direction, speed_multiplier: float = 1.0) -> None:
        """Move the player for one collision-safe fixed physics tick."""
        self._move(self.player, direction, speed_multiplier)
        vector = DIRECTION_VECTORS[direction]
        length = math.hypot(*vector)
        self.player_velocity = (
            vector[0] / length * self.player.speed * speed_multiplier,
            vector[1] / length * self.player.speed * speed_multiplier,
        )

    def _move(
        self, actor: Actor, direction: Direction, speed_multiplier: float
    ) -> None:
        vector = DIRECTION_VECTORS[direction]
        length = math.hypot(*vector)
        distance = actor.speed * speed_multiplier * PHYSICS_DT_SECONDS
        candidate = (
            actor.x + vector[0] / length * distance,
            actor.y + vector[1] / length * distance,
        )
        if not is_blocked(candidate, actor.radius, self.arena.bars):
            actor.x, actor.y = candidate
