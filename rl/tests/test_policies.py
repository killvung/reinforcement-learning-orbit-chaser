import numpy as np

from orbit_chase.rules import ACTION_COUNT, OBSERVATION_SIZE
from orbit_chase.observation import (
    COLLECTIBLE_FEATURE_WIDTH,
    ENEMY_POSITION_SLICE,
    PELLET_FEATURE_OFFSET,
)
from orbit_chase.policies import (
    EnemyEvadePolicy,
    PelletSeekingPolicy,
    Policy,
    RandomValidPolicy,
)


def _state() -> dict[str, np.ndarray]:
    return {
        "observation": np.zeros(OBSERVATION_SIZE, dtype=np.float32),
        "action_mask": np.ones(ACTION_COUNT, dtype=np.int8),
    }


def test_random_valid_policy_never_selects_a_masked_action():
    state = _state()
    state["action_mask"] = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)

    assert RandomValidPolicy().choose(state, np.random.default_rng(73)) == 2


def test_pellet_seeking_policy_uses_only_the_observation_and_action_mask():
    state = _state()
    state["observation"][
        PELLET_FEATURE_OFFSET : PELLET_FEATURE_OFFSET + COLLECTIBLE_FEATURE_WIDTH
    ] = [0.2, 0.0, 1.0]

    assert PelletSeekingPolicy().choose(state, np.random.default_rng(73)) == 2

    state["action_mask"][2] = 0
    assert PelletSeekingPolicy().choose(state, np.random.default_rng(73)) != 2


def test_enemy_evade_policy_moves_away_from_the_enemy():
    state = _state()
    state["observation"][ENEMY_POSITION_SLICE] = [0.5, 0.0]

    assert EnemyEvadePolicy().choose(state, np.random.default_rng(73)) == 6

    state["action_mask"][6] = 0
    assert EnemyEvadePolicy().choose(state, np.random.default_rng(73)) != 6


def test_policy_rejects_a_masked_action_from_select():
    class IllegalPolicy(Policy):
        name = "illegal"

        def select(self, state, valid, rng) -> int:
            return 0

    state = _state()
    state["action_mask"][0] = 0

    try:
        IllegalPolicy().choose(state, np.random.default_rng(73))
    except RuntimeError as error:
        assert str(error) == "illegal selected masked action 0."
    else:
        raise AssertionError("Policy.choose should reject a masked action")
