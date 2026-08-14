"""Characterization tests for the public Gymnasium environment contract."""

import numpy as np

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
    assert set(info) >= {"captured", "cleared", "pellets_collected"}


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
