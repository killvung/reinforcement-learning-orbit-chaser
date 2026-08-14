from types import SimpleNamespace

import numpy as np

from orbit_chase.arena import Arena, Collectible
from orbit_chase.rules import OBSERVATION_SIZE, PELLET_COUNT, SURGE_ORB_COUNT
from orbit_chase.observation import (
    BAR_FEATURE_COUNT,
    COLLECTIBLE_FEATURE_WIDTH,
    ENEMY_FEATURE_COUNT,
    PELLET_FEATURE_OFFSET,
    PLAYER_FEATURE_COUNT,
    PLAYER_POSITION_SLICE,
    TIME_FEATURE_COUNT,
    decode,
    encode,
)
from orbit_chase.simulation import GameSimulation


def make_snapshot():
    """Return a complete, fixed state with deliberately distinct test values."""
    pellets = tuple(Collectible((400 + index, 322 - index)) for index in range(32))
    orbs = tuple(Collectible((410 + index, 312 - index)) for index in range(3))
    arena = Arena(
        bars=(((300.0, 200.0), (350.0, 200.0)), ((450.0, 400.0), (500.0, 400.0))),
        pellet_slots=pellets,
        orb_slots=orbs,
    )
    return SimpleNamespace(
        arena=arena,
        player=SimpleNamespace(x=400.0, y=322.0),
        player_velocity=(175.0, -87.5),
        surge_remaining=2.0,
        enemy=SimpleNamespace(x=521.0, y=322.0),
        enemy_action=2,
        enemy_remaining=0.14,
        pellet_active=[True, False] + [True] * 30,
        orb_active=[False, True, True],
        time_remaining=30.0,
    )


def test_encode_has_documented_length_and_actor_features():
    observation = encode(make_snapshot())

    assert observation.shape == (OBSERVATION_SIZE,)
    np.testing.assert_allclose(observation[0:5], (0.0, 0.0, 1.0, -0.5, 0.5))
    np.testing.assert_allclose(observation[5:7], (0.5, 0.0))
    np.testing.assert_array_equal(observation[7:15], (0, 0, 1, 0, 0, 0, 0, 0))
    assert observation[15] == 0.5


def test_encode_preserves_slot_order_active_flags_and_time():
    observation = encode(make_snapshot())

    # First pellet occupies 24–26; second pellet's active flag is index 29.
    np.testing.assert_allclose(observation[24:27], (0.0, 0.0, 1.0))
    assert observation[29] == 0.0
    # First orb active flag is index 122; remaining-time fraction is last.
    assert observation[122] == 0.0
    assert observation[129] == 0.5


def test_decode_reads_player_enemy_and_active_pellet_positions():
    observation = encode(make_snapshot())
    view = decode(observation)

    np.testing.assert_allclose(view.player, (0.0, 0.0))
    np.testing.assert_allclose(view.enemy, (0.5, 0.0))
    assert len(view.pellets) == 31
    np.testing.assert_allclose(view.pellets[0], (0.0, 0.0))
    assert view.time_fraction == 0.5


def test_observation_layout_constants_match_the_documented_vector():
    assert PLAYER_POSITION_SLICE == slice(0, 2)
    assert PELLET_FEATURE_OFFSET == 24
    assert (
        PLAYER_FEATURE_COUNT
        + ENEMY_FEATURE_COUNT
        + BAR_FEATURE_COUNT
        + PELLET_COUNT * COLLECTIBLE_FEATURE_WIDTH
        + SURGE_ORB_COUNT * COLLECTIBLE_FEATURE_WIDTH
        + TIME_FEATURE_COUNT
        == OBSERVATION_SIZE
    )


def test_encode_accepts_real_game_simulation_state():
    observation = encode(GameSimulation(73))

    assert observation.shape == (OBSERVATION_SIZE,)
