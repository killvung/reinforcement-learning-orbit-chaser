from orbit_chase.simulation import GameSimulation
from parity.fixture import apply_setup, fixture_actions


def test_fixture_actions_expands_compact_segments_and_legacy_actions():
    assert fixture_actions({"segments": [{"direction": "up-right", "ticks": 2}]}) == [
        "up-right",
        "up-right",
    ]
    assert fixture_actions({"actions": ["left"]}) == ["left"]


def test_fixture_setup_replaces_slots_and_applies_explicit_state():
    simulation = GameSimulation(73)
    apply_setup(
        simulation,
        {
            "player": [250, 500],
            "enemy": [550, 100],
            "time_remaining": 0.005,
            "surge_remaining": 0.01,
            "pellet_slots": [[250, 500]],
            "orb_slots": [[251, 500]],
            "pellet_active": [False],
            "orb_active": [True],
        },
    )

    assert (simulation.player.x, simulation.player.y) == (250, 500)
    assert (simulation.enemy.x, simulation.enemy.y) == (550, 100)
    assert simulation.time_remaining == 0.005
    assert simulation.surge_remaining == 0.01
    assert simulation.arena.pellet_slots[0].position == (250, 500)
    assert simulation.pellet_active == [False]
    assert simulation.orb_active == [True]


def test_fixture_setup_activates_replacement_slots_when_flags_are_omitted():
    simulation = GameSimulation(73)
    apply_setup(simulation, {"pellet_slots": [[250, 500]], "orb_slots": []})

    assert simulation.pellet_active == [True]
    assert simulation.orb_active == []


def test_fixture_actions_rejects_unknown_directions_and_invalid_tick_counts():
    for fixture in (
        {"segments": [{"direction": "north", "ticks": 1}]},
        {"segments": [{"direction": "up", "ticks": 0}]},
    ):
        try:
            fixture_actions(fixture)
        except ValueError:
            continue
        raise AssertionError("Invalid fixture was accepted")
