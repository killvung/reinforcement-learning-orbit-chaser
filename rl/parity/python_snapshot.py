"""Emit deterministic snapshots for the initial movement parity fixture."""

from __future__ import annotations

import json
from pathlib import Path

from orbit_chase.constants import BAR_WIDTH, Direction, ENEMY_DECISION_SECONDS
from orbit_chase.simulation import GameSimulation, StepEvents

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "movement.json"


def snapshot(simulation: GameSimulation, tick: int, events: StepEvents) -> dict:
    point = lambda value: {"x": value[0], "y": value[1]}
    return {
        "tick": tick,
        "player": {"x": simulation.player.x, "y": simulation.player.y},
        "enemy": {"x": simulation.enemy.x, "y": simulation.enemy.y},
        "enemy_direction": simulation.enemy_action.name.lower().replace("_", "-"),
        "enemy_decision_fraction": max(0.0, simulation.enemy_remaining) / ENEMY_DECISION_SECONDS,
        "surge_remaining": simulation.surge_remaining,
        "time_remaining": simulation.time_remaining,
        "bars": [
            {"from": point(start), "to": point(end), "width": BAR_WIDTH}
            for start, end in simulation.arena.bars
        ],
        "pellet_slots": [point(slot.position) for slot in simulation.arena.pellet_slots],
        "orb_slots": [point(slot.position) for slot in simulation.arena.orb_slots],
        "pellet_active": simulation.pellet_active,
        "orb_active": simulation.orb_active,
        "events": {"captured": events.captured, "cleared": events.cleared, "timed_out": events.timed_out},
    }


def main() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    simulation, events, snapshots = GameSimulation(fixture["seed"]), StepEvents(), []
    for tick in range(len(fixture["actions"]) + 1):
        if tick in fixture["snapshot_ticks"]:
            snapshots.append(snapshot(simulation, tick, events))
        if tick < len(fixture["actions"]):
            events = simulation.step(Direction[fixture["actions"][tick].upper()])
    print(json.dumps({"seed": fixture["seed"], "snapshots": snapshots}))


if __name__ == "__main__":
    main()
