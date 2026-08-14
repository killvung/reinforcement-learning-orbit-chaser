"""Characterization tests for the public Gymnasium environment contract."""

import numpy as np

from orbit_chase_player_env import OrbitChasePlayerEnv


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
