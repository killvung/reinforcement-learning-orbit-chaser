"""Characterization tests for the public Gymnasium environment contract."""

import math

import numpy as np

from orbit_chase.constants import PHYSICS_DT_SECONDS
from orbit_chase_player_env import OrbitChasePlayerEnv
from orbit_chase.simulation import StepEvents


def test_reset_with_same_seed_returns_same_observation():
    first = OrbitChasePlayerEnv()
    second = OrbitChasePlayerEnv()
    first_observation, _ = first.reset(seed=73)
    second_observation, _ = second.reset(seed=73)

    np.testing.assert_array_equal(
        first_observation["observation"],
        second_observation["observation"],
    )
    np.testing.assert_array_equal(
        first_observation["action_mask"],
        second_observation["action_mask"],
    )


def test_step_returns_gymnasium_transition_shape():
    environment = OrbitChasePlayerEnv()
    environment.reset(seed=73)

    observation, reward, terminated, truncated, info = environment.step(2)

    assert observation["observation"].shape == (130,)
    assert observation["action_mask"].shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info) >= {
        "captured",
        "cleared",
        "timed_out",
        "pellets_collected",
        "orbs_collected",
    }


def test_step_accumulates_pickups_across_the_full_decision_interval():
    """A pickup on tick one must affect the reward after all ten ticks finish."""

    class SimulationWithEarlyPellet:
        def __init__(self):
            self.ticks = 0

        def step(self, direction):
            self.ticks += 1
            return StepEvents(pellets_collected=1 if self.ticks == 1 else 0)

    environment = OrbitChasePlayerEnv()
    environment.simulation = SimulationWithEarlyPellet()
    environment._observation = lambda: {}

    _, reward, _, _, info = environment.step(2)

    assert reward == 2.99
    assert info["pellets_collected"] == 1


def test_capture_reward_takes_priority_over_simultaneous_clear():
    """Browser gameplay declares capture first when both terminal events occur."""
    events = StepEvents(captured=True, cleared=True)

    assert OrbitChasePlayerEnv._reward(events) == -100.01


def test_step_advances_exactly_one_player_decision_interval():
    environment = OrbitChasePlayerEnv()
    environment.reset(seed=73)
    original_step = environment.simulation.step
    ticks = 0

    def count_ticks(direction):
        nonlocal ticks
        ticks += 1
        return original_step(direction)

    environment.simulation.step = count_ticks
    environment.step(2)

    assert ticks == 10
    assert math.isclose(
        environment.simulation.time_remaining,
        60.0 - 10 * PHYSICS_DT_SECONDS,
    )


def test_real_pellet_collection_updates_reward_info_and_slot_state():
    environment = OrbitChasePlayerEnv()
    environment.reset(seed=73)
    simulation = environment.simulation
    simulation.pellet_active = [False] * len(simulation.pellet_active)
    simulation.pellet_active[0:2] = [True, True]
    simulation.player.x, simulation.player.y = simulation.arena.pellet_slots[0].position

    _, reward, _, _, info = environment.step(0)

    assert reward == 2.99
    assert info["pellets_collected"] == 1
    assert simulation.pellet_active[0] is False


def test_terminal_events_map_to_gymnasium_flags():
    capture = OrbitChasePlayerEnv()
    capture.reset(seed=73)
    capture.simulation.player.x, capture.simulation.player.y = (
        capture.simulation.enemy.x,
        capture.simulation.enemy.y,
    )
    _, _, terminated, truncated, _ = capture.step(0)
    assert terminated is True
    assert truncated is False

    clear = OrbitChasePlayerEnv()
    clear.reset(seed=73)
    clear.simulation.pellet_active = [False] * len(clear.simulation.pellet_active)
    _, _, terminated, truncated, _ = clear.step(0)
    assert terminated is True
    assert truncated is False

    timeout = OrbitChasePlayerEnv()
    timeout.reset(seed=73)
    timeout.simulation.time_remaining = PHYSICS_DT_SECONDS / 2
    _, _, terminated, truncated, _ = timeout.step(0)
    assert terminated is False
    assert truncated is True


def test_returned_state_satisfies_declared_gymnasium_space():
    environment = OrbitChasePlayerEnv()
    state, _ = environment.reset(seed=73)
    assert environment.observation_space.contains(state)

    state, *_ = environment.step(2)
    assert environment.observation_space.contains(state)


def test_terminal_precedence_uses_terminated_for_capture_or_clear():
    class TerminalSimulation:
        def __init__(self, events):
            self.events = events

        def step(self, direction):
            return self.events

    for events in (
        StepEvents(captured=True, timed_out=True),
        StepEvents(cleared=True, timed_out=True),
    ):
        environment = OrbitChasePlayerEnv()
        environment.simulation = TerminalSimulation(events)
        environment._observation = lambda: {}
        _, _, terminated, truncated, _ = environment.step(2)
        assert terminated is True
        assert truncated is False


def test_terminal_event_stops_decision_interval_early():
    class CaptureOnThirdTick:
        def __init__(self):
            self.ticks = 0

        def step(self, direction):
            self.ticks += 1
            return StepEvents(captured=self.ticks == 3)

    environment = OrbitChasePlayerEnv()
    environment.simulation = CaptureOnThirdTick()
    environment._observation = lambda: {}

    environment.step(2)

    assert environment.simulation.ticks == 3


def test_step_rewards_multiple_pickups_from_one_decision():
    class MultiPickupSimulation:
        def __init__(self):
            self.ticks = 0

        def step(self, direction):
            self.ticks += 1
            return StepEvents(
                pellets_collected=2 if self.ticks == 1 else 0,
                orbs_collected=1 if self.ticks == 1 else 0,
            )

    environment = OrbitChasePlayerEnv()
    environment.simulation = MultiPickupSimulation()
    environment._observation = lambda: {}

    _, reward, _, _, info = environment.step(2)

    assert reward == 15.99
    assert info["pellets_collected"] == 2
    assert info["orbs_collected"] == 1
