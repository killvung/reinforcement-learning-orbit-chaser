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
    ENEMY_DECISION_SECONDS,
    PHYSICS_DT_SECONDS,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    PLAYER_START,
    PELLET_COLLECTION_RADIUS,
    Direction,
    ORB_COLLECTION_RADIUS,
    ROUND_DURATION_SECONDS,
    SURGE_DURATION_SECONDS,
    ENEMY_SURGE_SPEED_MULTIPLIER,
    PLAYER_SURGE_SPEED_MULTIPLIER,
)
from .geometry import distance, is_blocked


@dataclass
class Actor:
    """Mutable circular actor state used by the physics simulation."""

    x: float
    y: float
    radius: float
    speed: float


@dataclass(frozen=True)
class StepEvents:
    """Facts produced by one physics tick; rewards belong in the Gym adapter."""

    pellets_collected: int = 0
    orbs_collected: int = 0
    captured: bool = False
    cleared: bool = False
    timed_out: bool = False


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
        self.pellet_active = [True] * len(self.arena.pellet_slots)
        self.orb_active = [True] * len(self.arena.orb_slots)
        self.surge_remaining = 0.0
        self.time_remaining = ROUND_DURATION_SECONDS
        self.enemy_action = Direction.LEFT
        self.enemy_remaining = 0.0

    def move_player(self, direction: Direction, speed_multiplier: float = 1.0) -> None:
        """Move the player for one collision-safe fixed physics tick."""
        self._move(self.player, direction, speed_multiplier)
        vector = DIRECTION_VECTORS[direction]
        length = math.hypot(*vector)
        self.player_velocity = (
            vector[0] / length * self.player.speed * speed_multiplier,
            vector[1] / length * self.player.speed * speed_multiplier,
        )

    def step(self, player_direction: Direction) -> StepEvents:
        """Advance exactly one fixed tick of player and deterministic enemy rules."""
        surge_is_active = self._advance_surge_clock()
        self.move_player(
            player_direction,
            PLAYER_SURGE_SPEED_MULTIPLIER if surge_is_active else 1.0,
        )
        self._advance_enemy(surge_is_active)
        pellets_collected, orbs_collected = self._collect_pickups()
        self.time_remaining -= PHYSICS_DT_SECONDS
        return self._step_events(pellets_collected, orbs_collected)

    def _advance_surge_clock(self) -> bool:
        """Expire Surge before movement, matching the browser simulation."""
        self.surge_remaining = max(0.0, self.surge_remaining - PHYSICS_DT_SECONDS)
        return self.surge_remaining > 0

    def _advance_enemy(self, surge_is_active: bool) -> None:
        """Choose an action when due, then move the enemy for one fixed tick."""
        if self.enemy_remaining <= 0:
            self.enemy_action = self._choose_enemy_action()
            self.enemy_remaining = ENEMY_DECISION_SECONDS
        speed_multiplier = ENEMY_SURGE_SPEED_MULTIPLIER if surge_is_active else 1.0
        self._move(self.enemy, self.enemy_action, speed_multiplier)
        self.enemy_remaining -= PHYSICS_DT_SECONDS

    def _collect_pickups(self) -> tuple[int, int]:
        """Collect nearby pellets and Orbs, starting Surge when an Orb is found."""
        pellets = self._collect(
            self.arena.pellet_slots,
            self.pellet_active,
            PELLET_COLLECTION_RADIUS,
        )
        orbs = self._collect(self.arena.orb_slots, self.orb_active, ORB_COLLECTION_RADIUS)
        if orbs:
            self.surge_remaining = SURGE_DURATION_SECONDS
        return pellets, orbs

    def _step_events(self, pellets_collected: int, orbs_collected: int) -> StepEvents:
        """Summarize this tick's collections and terminal conditions."""
        captured = distance(
            (self.player.x, self.player.y),
            (self.enemy.x, self.enemy.y),
        ) < self.player.radius + self.enemy.radius
        return StepEvents(
            pellets_collected=pellets_collected,
            orbs_collected=orbs_collected,
            captured=captured,
            cleared=not any(self.pellet_active),
            timed_out=self.time_remaining <= 0,
        )

    def _collect(self, slots, active_flags, collection_radius: float) -> int:
        collected = 0
        for index, slot in enumerate(slots):
            if active_flags[index] and distance((self.player.x, self.player.y), slot.position) <= collection_radius:
                active_flags[index] = False
                collected += 1
        return collected

    def _choose_enemy_action(self) -> Direction:
        choices = [direction for direction in Direction if self._can_travel(self.enemy, direction, ENEMY_DECISION_SECONDS)]
        return min(choices or [self.enemy_action], key=lambda direction: distance((self.enemy.x + DIRECTION_VECTORS[direction][0] * 42, self.enemy.y + DIRECTION_VECTORS[direction][1] * 42), (self.player.x, self.player.y)))

    def _can_travel(self, actor: Actor, direction: Direction, duration: float) -> bool:
        vector, length = DIRECTION_VECTORS[direction], math.hypot(*DIRECTION_VECTORS[direction])
        distance_to_travel = actor.speed * duration
        steps = max(1, math.ceil(distance_to_travel / 4))
        return all(not is_blocked((actor.x + vector[0] / length * distance_to_travel * index / steps, actor.y + vector[1] / length * distance_to_travel * index / steps), actor.radius, self.arena.bars) for index in range(1, steps + 1))

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
