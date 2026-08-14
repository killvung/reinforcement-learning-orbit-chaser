"""Deterministic arena and collectible generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    ARENA_CENTER,
    ACTOR_SPAWN_CLEARANCE,
    BAR_LENGTH_VARIATION,
    BAR_MIN_LENGTH,
    BAR_OFFSET,
    COLLECTIBLE_SEED_SALT,
    ENEMY_START,
    ORB_SPAWN_MIN_RADIUS,
    ORB_SPAWN_RADIUS_VARIATION,
    PELLET_COUNT,
    PELLET_SPAWN_MIN_RADIUS,
    PELLET_SPAWN_RADIUS_VARIATION,
    PELLET_SPAWN_BODY_RADIUS,
    ORB_SPAWN_BODY_RADIUS,
    ORB_TO_PELLET_CLEARANCE,
    PLAYER_START,
    SURGE_ORB_COUNT,
)
from .geometry import distance, is_blocked


Point = tuple[float, float]
Bar = tuple[Point, Point]


@dataclass(frozen=True)
class Collectible:
    """A fixed spawn slot whose active state belongs to an episode."""

    position: Point


@dataclass(frozen=True)
class Arena:
    """Static geometry and collectible spawn slots generated from one seed."""

    bars: tuple[Bar, Bar]
    pellet_slots: tuple[Collectible, ...]
    orb_slots: tuple[Collectible, ...]


def mulberry32(seed: int):
    """Return the unsigned 32-bit PRNG used by the TypeScript arena generator."""
    value = seed & 0xFFFFFFFF

    def random() -> float:
        nonlocal value
        value = (value + 0x6D2B79F5) & 0xFFFFFFFF
        result = value
        result = ((result ^ (result >> 15)) * (result | 1)) & 0xFFFFFFFF
        result ^= (
            result + (((result ^ (result >> 7)) * (result | 61)) & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        return ((result ^ (result >> 14)) & 0xFFFFFFFF) / 4294967296

    return random


def make_arena(seed: int) -> Arena:
    """Create the exact static arena and collectible slots for a supplied seed."""
    bars = _make_bars(mulberry32(seed))
    spawn_random = mulberry32(seed ^ COLLECTIBLE_SEED_SALT)
    pellets = _make_pellet_slots(spawn_random, bars)
    orbs = _make_orb_slots(spawn_random, bars, pellets)
    return Arena(bars, tuple(pellets), tuple(orbs))


def _make_bars(random) -> tuple[Bar, Bar]:
    """Generate the two bars with a shared orientation."""
    angle = random() * math.tau
    unit, perpendicular = (math.cos(angle), math.sin(angle)), (
        -math.sin(angle),
        math.cos(angle),
    )
    bars: list[Bar] = []
    for sign in (-1, 1):
        offset = BAR_OFFSET * sign
        length = BAR_MIN_LENGTH + random() * BAR_LENGTH_VARIATION
        bars.append(
            (
                (
                    ARENA_CENTER[0] + unit[0] * offset - perpendicular[0] * length / 2,
                    ARENA_CENTER[1] + unit[1] * offset - perpendicular[1] * length / 2,
                ),
                (
                    ARENA_CENTER[0] + unit[0] * offset + perpendicular[0] * length / 2,
                    ARENA_CENTER[1] + unit[1] * offset + perpendicular[1] * length / 2,
                ),
            )
        )
    return bars[0], bars[1]


def _make_pellet_slots(random, bars: tuple[Bar, Bar]) -> list[Collectible]:
    """Generate pellet slots that avoid actors and obstacles."""
    pellets: list[Collectible] = []
    while len(pellets) < PELLET_COUNT:
        point = _spawn(random, PELLET_SPAWN_MIN_RADIUS, PELLET_SPAWN_RADIUS_VARIATION)
        if (
            not is_blocked(point, PELLET_SPAWN_BODY_RADIUS, bars)
            and distance(point, PLAYER_START) > ACTOR_SPAWN_CLEARANCE
            and distance(point, ENEMY_START) > ACTOR_SPAWN_CLEARANCE
        ):
            pellets.append(Collectible(point))
    return pellets


def _make_orb_slots(
    random, bars: tuple[Bar, Bar], pellets: list[Collectible]
) -> list[Collectible]:
    """Generate Surge Orb slots that do not overlap pellet slots."""
    orbs: list[Collectible] = []
    while len(orbs) < SURGE_ORB_COUNT:
        point = _spawn(random, ORB_SPAWN_MIN_RADIUS, ORB_SPAWN_RADIUS_VARIATION)
        if not is_blocked(point, ORB_SPAWN_BODY_RADIUS, bars) and all(
            distance(point, slot.position) > ORB_TO_PELLET_CLEARANCE for slot in pellets
        ):
            orbs.append(Collectible(point))
    return orbs


def _spawn(random, low: float, spread: float) -> Point:
    angle, radius = random() * math.tau, low + random() * spread
    return (
        ARENA_CENTER[0] + math.cos(angle) * radius,
        ARENA_CENTER[1] + math.sin(angle) * radius,
    )
