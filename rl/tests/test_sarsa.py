import numpy as np

from orbit_chase.agent import Transition
from orbit_chase.rules import ACTION_COUNT, OBSERVATION_SIZE
from orbit_chase.sarsa import (
    FEATURE_GROUP_SALTS,
    GROUP_ORB_RANK,
    GROUP_PELLET_RANK,
    FeatureEncoder,
    LinearSarsaAgent,
    SarsaConfig,
    SparseFeatures,
)


def _state() -> dict[str, np.ndarray]:
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    observation[0:5] = [0.1, -0.2, 0.3, -0.4, 0.5]
    observation[5:7] = [-0.3, 0.2]
    observation[7] = 1.0
    observation[15] = 0.75
    observation[16:24] = [-0.5, -0.3, -0.1, -0.3, 0.1, 0.2, 0.5, 0.2]
    observation[24:30] = [0.2, 0.1, 1.0, -0.4, 0.2, 1.0]
    observation[120:126] = [0.4, -0.1, 1.0, -0.1, -0.4, 1.0]
    observation[129] = 0.9
    return {
        "observation": observation,
        "action_mask": np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8),
    }


def test_feature_encoder_is_deterministic_sorted_and_bounded():
    encoder = FeatureEncoder(capacity=257)

    first = encoder.encode(_state())
    second = encoder.encode(_state())

    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.values, second.values)
    assert np.all(first.indices[:-1] < first.indices[1:])
    assert np.all(first.indices >= 0)
    assert np.all(first.indices < 257)
    assert np.all(first.values > 0.0)


def test_feature_encoder_uses_action_mask_as_public_state():
    encoder = FeatureEncoder(capacity=65_536)
    open_state = _state()
    blocked_state = _state()
    blocked_state["action_mask"][2] = 0

    assert not np.array_equal(
        encoder.encode(open_state).indices,
        encoder.encode(blocked_state).indices,
    )


def test_empty_pellet_rank_is_not_a_pellet_on_the_player():
    encoder = FeatureEncoder(capacity=65_536)
    empty = _state()
    empty["observation"][24:120] = 0.0
    on_player = _state()
    on_player["observation"][24:120] = 0.0
    on_player["observation"][24:27] = [
        on_player["observation"][0],
        on_player["observation"][1],
        1.0,
    ]

    assert not np.array_equal(
        encoder.encode(empty).indices,
        encoder.encode(on_player).indices,
    )


def test_tile_bins_stay_inside_the_documented_range():
    encoder = FeatureEncoder(capacity=65_536, bins=8, tilings=8)
    lower = set()
    upper = set()
    encoder._add_tiles(lower, 99, [-2.0, -2.0], -2.0, 2.0)
    encoder._add_tiles(upper, 99, [2.0, 2.0], -2.0, 2.0)
    just_inside = set()
    encoder._add_tiles(just_inside, 99, [2.0 - 1e-9, 2.0 - 1e-9], -2.0, 2.0)

    assert lower != upper
    assert upper == just_inside


def test_feature_group_salts_are_unique():
    assert len(FEATURE_GROUP_SALTS) == len(set(FEATURE_GROUP_SALTS))

    encoder = FeatureEncoder(capacity=65_536)
    pellet = set()
    orb = set()
    encoder._add_tiles(pellet, GROUP_PELLET_RANK, [0.1, 0.2], -2.0, 2.0)
    encoder._add_tiles(orb, GROUP_ORB_RANK, [0.1, 0.2], -2.0, 2.0)
    assert pellet != orb


def test_masked_greedy_uses_lowest_valid_tie_break_and_exploration_stays_valid():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=32), SarsaConfig(epsilon=0.0), seed=73)
    values = np.zeros(ACTION_COUNT)
    values[2] = 4.0
    values[3] = 4.0
    mask = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int8)

    assert agent._select_from_q(values, mask, epsilon=0.0) == 2

    exploratory = LinearSarsaAgent(
        FeatureEncoder(capacity=32), SarsaConfig(epsilon=1.0), seed=73
    )
    assert {exploratory._select_from_q(values, mask, epsilon=1.0) for _ in range(40)} <= {
        2,
        3,
    }


def test_choose_is_greedy_and_does_not_advance_the_training_rng():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=4096), SarsaConfig(epsilon=1.0), seed=73)
    state = _state()
    features = agent.encoder.encode(state)
    agent.weights[2, features.indices] = 10.0
    rng_before = agent.rng.bit_generator.state

    for _ in range(20):
        assert agent.choose(state, np.random.default_rng(1)) == 2

    assert agent.rng.bit_generator.state == rng_before


def test_true_online_update_matches_hand_computed_first_transition():
    agent = LinearSarsaAgent(
        FeatureEncoder(capacity=16),
        SarsaConfig(alpha=0.25, gamma=0.995, lambda_=0.9),
        seed=73,
    )
    features = SparseFeatures(
        indices=np.array([2, 7], dtype=np.int32), values=np.array([1.0, 1.0])
    )

    agent.update_features(features, action=3, reward=4.0, terminal=True)

    assert agent.q_value(features, 3) == 2.0
    assert agent.weights[3, 2] == 1.0
    assert agent.weights[3, 7] == 1.0
    assert agent.q_old == 0.0


def test_lambda_zero_matches_one_step_sarsa_update():
    config = SarsaConfig(alpha=0.5, gamma=0.9, lambda_=0.0)
    agent = LinearSarsaAgent(FeatureEncoder(capacity=16), config, seed=73)
    first = SparseFeatures(
        indices=np.array([1], dtype=np.int32), values=np.array([1.0])
    )
    second = SparseFeatures(
        indices=np.array([2], dtype=np.int32), values=np.array([1.0])
    )
    agent.weights[4, 2] = 6.0

    agent.update_features(first, action=3, reward=2.0, next_features=second, next_action=4)

    # delta = 2 + 0.9 * 6 - 0 = 7.4; with lambda = 0, w += alpha * delta * x.
    assert agent.weights[3, 1] == 3.7
    assert agent.q_old == 6.0


def test_true_online_trace_correction_uses_the_trace_before_decay():
    agent = LinearSarsaAgent(
        FeatureEncoder(capacity=16),
        SarsaConfig(alpha=0.25, gamma=0.5, lambda_=0.5),
        seed=73,
    )
    features = SparseFeatures(
        indices=np.array([1], dtype=np.int32), values=np.array([1.0])
    )

    agent.update_features(features, action=3, reward=4.0, terminal=True)
    agent.update_features(features, action=3, reward=0.0, terminal=True)

    # e <- 0.25 * 1 + (1 - 0.25 * 0.5 * 0.5 * 1) * 1 = 1.1875.
    assert agent.traces[(3, 1)] == 1.1875


def test_terminal_update_does_not_read_a_next_action_value():
    agent = LinearSarsaAgent(
        FeatureEncoder(capacity=16), SarsaConfig(alpha=0.5, lambda_=0.0), seed=73
    )
    features = SparseFeatures(
        indices=np.array([1], dtype=np.int32), values=np.array([1.0])
    )
    agent.weights[:, 1] = 100.0

    agent.update_features(features, action=3, reward=2.0, terminal=True)

    assert agent.q_old == 0.0
    assert agent.weights[3, 1] == 51.0


def test_gym_transition_update_encodes_state():
    agent = LinearSarsaAgent(
        FeatureEncoder(capacity=4096), SarsaConfig(alpha=0.5, lambda_=0.0), seed=73
    )
    state = _state()
    delta = agent.update(
        Transition(
            state=state,
            action=2,
            reward=1.0,
            next_state=state,
            next_action=None,
            terminal=True,
        )
    )

    assert delta == 1.0
    assert np.count_nonzero(agent.weights) > 0


def test_feature_encoder_rejects_invalid_public_state_shapes():
    encoder = FeatureEncoder()
    state = _state()
    state["observation"] = np.zeros(OBSERVATION_SIZE - 1, dtype=np.float32)

    try:
        encoder.encode(state)
    except ValueError as error:
        assert "observation" in str(error)
    else:
        raise AssertionError("FeatureEncoder.encode should validate observation shape")


def test_feature_encoder_rejects_nonbinary_action_masks():
    state = _state()
    state["action_mask"][0] = 2

    try:
        FeatureEncoder().encode(state)
    except ValueError as error:
        assert "binary" in str(error)
    else:
        raise AssertionError("FeatureEncoder.encode should validate action mask values")
