"""Gym-free, deterministic fixed-tick Orbit Chase game rules."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .arena import Arena, make_arena
from .constants import (
    DIRECTION_UNITS,
    DIRECTION_VECTORS,
    ENEMY_LOOKAHEAD_PIXELS,
    ENEMY_RADIUS,
    ENEMY_SPEED,
    ENEMY_START,
    ENEMY_DECISION_SECONDS,
    PATH_COLLISION_SEGMENT_PIXELS,
    PHYSICS_DT_SECONDS,
    PLAYER_RADIUS,
    PLAYER_DECISION_SECONDS,
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
        dx, dy = DIRECTION_UNITS[direction]
        speed = self.player.speed * speed_multiplier
        self.player_velocity = (dx * speed, dy * speed)

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

    def valid_actions(self) -> list[bool]:
        """Return collision-safe player directions for the next 100 ms decision."""
        speed_multiplier = (
            PLAYER_SURGE_SPEED_MULTIPLIER if self.surge_remaining > 0 else 1.0
        )
        return [
            self._can_travel(
                self.player,
                direction,
                PLAYER_DECISION_SECONDS,
                speed_multiplier,
            )
            for direction in Direction
        ]

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
        orbs = self._collect(
            self.arena.orb_slots, self.orb_active, ORB_COLLECTION_RADIUS
        )
        if orbs:
            self.surge_remaining = SURGE_DURATION_SECONDS
        return pellets, orbs

    def _step_events(self, pellets_collected: int, orbs_collected: int) -> StepEvents:
        """Summarize this tick's collections and terminal conditions."""
        captured = (
            distance(
                (self.player.x, self.player.y),
                (self.enemy.x, self.enemy.y),
            )
            < self.player.radius + self.enemy.radius
        )
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
            if (
                active_flags[index]
                and distance((self.player.x, self.player.y), slot.position)
                <= collection_radius
            ):
                active_flags[index] = False
                collected += 1
        return collected

    def _choose_enemy_action(self) -> Direction:
        choices = [
            direction
            for direction in Direction
            if self._can_travel(self.enemy, direction, ENEMY_DECISION_SECONDS)
        ]

        def lookahead_distance(direction: Direction) -> float:
            dx, dy = DIRECTION_VECTORS[direction]
            return distance(
                (
                    self.enemy.x + dx * ENEMY_LOOKAHEAD_PIXELS,
                    self.enemy.y + dy * ENEMY_LOOKAHEAD_PIXELS,
                ),
                (self.player.x, self.player.y),
            )

        return min(choices or [self.enemy_action], key=lookahead_distance)

    def _can_travel(
        self,
        actor: Actor,
        direction: Direction,
        duration: float,
        speed_multiplier: float = 1.0,
    ) -> bool:
        """Check every short segment of a proposed path for obstacle overlap.

        Testing the full swept path prevents an actor from crossing a narrow
        bar when only its final destination would appear collision-free.
        """
        # Resolve the requested direction into a unit movement vector.
        dx, dy = DIRECTION_UNITS[direction]

        # Divide the requested distance into short collision-check segments.
        travel = actor.speed * speed_multiplier * duration
        steps = max(1, math.ceil(travel / PATH_COLLISION_SEGMENT_PIXELS))
        step_length = travel / steps

        # Sweep every candidate position; the first obstacle blocks the path.
        for step_index in range(1, steps + 1):
            candidate = (
                actor.x + dx * step_length * step_index,
                actor.y + dy * step_length * step_index,
            )
            if is_blocked(candidate, actor.radius, self.arena.bars):
                return False

        return True

    def _move(
        self, actor: Actor, direction: Direction, speed_multiplier: float
    ) -> None:
        dx, dy = DIRECTION_UNITS[direction]
        travel = actor.speed * speed_multiplier * PHYSICS_DT_SECONDS
        candidate = (
            actor.x + dx * travel,
            actor.y + dy * travel,
        )
        if not is_blocked(candidate, actor.radius, self.arena.bars):
            actor.x, actor.y = candidate
