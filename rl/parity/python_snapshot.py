"""Emit deterministic snapshots for the initial movement parity fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from orbit_chase.constants import BAR_WIDTH, Direction, ENEMY_DECISION_SECONDS
from orbit_chase.simulation import GameSimulation, StepEvents
from parity.fixture import apply_setup, fixture_actions

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "movement.json"


def snapshot(simulation: GameSimulation, tick: int, events: StepEvents) -> dict:
    point = lambda value: {"x": value[0], "y": value[1]}
    return {
        "tick": tick,
        "player": {"x": simulation.player.x, "y": simulation.player.y},
        "player_velocity": {"x": simulation.player_velocity[0], "y": simulation.player_velocity[1]},
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
        # Copy mutable flags so a later collection cannot alter this snapshot.
        "pellet_active": list(simulation.pellet_active),
        "orb_active": list(simulation.orb_active),
        "events": {
            "pellets_collected": events.pellets_collected,
            "orbs_collected": events.orbs_collected,
            "captured": events.captured,
            "cleared": events.cleared,
            "timed_out": events.timed_out,
        },
    }


def main() -> None:
    fixture_path = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE_PATH
    fixture = json.loads(fixture_path.read_text())
    actions = fixture_actions(fixture)
    simulation, events, snapshots = GameSimulation(fixture["seed"]), StepEvents(), []
    apply_setup(simulation, fixture.get("setup"))
    for tick in range(len(actions) + 1):
        if tick in fixture["snapshot_ticks"]:
            snapshots.append(snapshot(simulation, tick, events))
        if tick < len(actions):
            events = simulation.step(Direction[actions[tick].upper().replace("-", "_")])
    print(json.dumps({"seed": fixture["seed"], "snapshots": snapshots}))


if __name__ == "__main__":
    main()
