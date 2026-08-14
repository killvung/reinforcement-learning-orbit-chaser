"""Shared Python fixture parsing and deterministic setup helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orbit_chase.arena import Arena, Collectible
from orbit_chase.rules import Direction
from orbit_chase.simulation import GameSimulation


FIXTURE_DIRECTIONS = frozenset(
    direction.name.lower().replace("_", "-") for direction in Direction
)


def fixture_actions(fixture: dict[str, Any]) -> list[str]:
    """Expand compact segments, while accepting the original action-array form."""
    segments = fixture.get("segments")
    if segments is not None:
        if not isinstance(segments, list):
            raise ValueError("Fixture segments must be an array.")
        return [
            direction
            for segment in segments
            for direction in _segment_actions(segment)
        ]

    actions = fixture.get("actions")
    if not isinstance(actions, list):
        raise ValueError("Fixture must contain actions or segments.")
    for direction in actions:
        _validate_direction(direction)
    return actions.copy()


def apply_setup(simulation: GameSimulation, setup: dict[str, Any] | None) -> None:
    """Apply only the explicit state overrides used by an event fixture."""
    if not setup:
        return
    if "player" in setup:
        simulation.player.x, simulation.player.y = _point(setup["player"])
    if "enemy" in setup:
        simulation.enemy.x, simulation.enemy.y = _point(setup["enemy"])
    if "time_remaining" in setup:
        simulation.time_remaining = float(setup["time_remaining"])
    if "surge_remaining" in setup:
        simulation.surge_remaining = float(setup["surge_remaining"])

    bars = _bars(setup.get("bars", simulation.arena.bars))
    pellets = _slots(setup.get("pellet_slots"), simulation.arena.pellet_slots)
    orbs = _slots(setup.get("orb_slots"), simulation.arena.orb_slots)
    simulation.arena = Arena(bars, pellets, orbs)
    simulation.pellet_active = _setup_active_flags(
        setup, "pellet_slots", "pellet_active", pellets, simulation.pellet_active
    )
    simulation.orb_active = _setup_active_flags(
        setup, "orb_slots", "orb_active", orbs, simulation.orb_active
    )


def _segment_actions(segment: Any) -> list[str]:
    if not isinstance(segment, dict):
        raise ValueError("Each fixture segment must be an object.")
    direction, ticks = segment.get("direction"), segment.get("ticks")
    _validate_direction(direction)
    if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 1:
        raise ValueError(f"Fixture ticks must be a positive integer: {ticks}")
    return [direction] * ticks


def _validate_direction(direction: Any) -> None:
    if direction not in FIXTURE_DIRECTIONS:
        raise ValueError(f"Unknown fixture direction: {direction}")


def _point(value: Any) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"Fixture point must contain two coordinates: {value}")
    return float(value[0]), float(value[1])


def _bars(value: Any) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("Fixture setup must contain exactly two bars.")
    return tuple((_point(start), _point(end)) for start, end in value)


def _slots(value: Any, existing: tuple[Collectible, ...]) -> tuple[Collectible, ...]:
    if value is None:
        return existing
    if not isinstance(value, list):
        raise ValueError("Fixture collectible slots must be an array.")
    return tuple(Collectible(_point(point)) for point in value)


def _active_flags(value: Any, slot_count: int) -> list[bool]:
    if not isinstance(value, list) or len(value) != slot_count or not all(
        isinstance(flag, bool) for flag in value
    ):
        raise ValueError("Fixture active flags must match their slot count and be boolean.")
    return value.copy()


def _setup_active_flags(
    setup: dict[str, Any],
    slots_key: str,
    active_key: str,
    slots: tuple[Collectible, ...],
    existing_flags: list[bool],
) -> list[bool]:
    """Use fresh active flags for replacement slots, otherwise preserve state."""
    if active_key in setup:
        flags = setup[active_key]
    elif slots_key in setup:
        flags = [True] * len(slots)
    else:
        flags = existing_flags
    return _active_flags(flags, len(slots))
