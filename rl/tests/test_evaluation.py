import evaluate_baselines
import numpy as np
import pytest

from orbit_chase.evaluation import _clear_time_seconds, evaluate_policy
from orbit_chase.policies import RandomValidPolicy
from orbit_chase.rules import ROUND_DURATION_SECONDS, TerminalOutcome
from orbit_chase.sarsa import FeatureEncoder, LinearSarsaAgent, SarsaConfig


def test_baseline_evaluation_is_reproducible_for_fixed_held_out_seeds():
    seeds = (10_000, 10_001)

    first = evaluate_policy(RandomValidPolicy(), seeds)
    second = evaluate_policy(RandomValidPolicy(), seeds)

    assert first == second
    assert first.episodes == len(seeds)


def test_clear_time_reads_the_simulation_clock():
    remaining = 51.099
    assert _clear_time_seconds(TerminalOutcome.CLEARED, remaining) == (
        ROUND_DURATION_SECONDS - remaining
    )
    assert _clear_time_seconds(TerminalOutcome.CAPTURED, remaining) is None
    assert _clear_time_seconds(TerminalOutcome.TIMEOUT, remaining) is None


def test_seed_split_label_matches_documented_ranges():
    assert evaluate_baselines.seed_split_label(8000, 100) == "validation"
    assert evaluate_baselines.seed_split_label(8000, 1000) == "validation"
    assert evaluate_baselines.seed_split_label(10000, 100) == "final-test"
    assert evaluate_baselines.seed_split_label(0, 8000) == "training"
    assert evaluate_baselines.seed_split_label(10000, 1000) == "custom"


def test_checkpoint_eval_requires_explicit_start_seed():
    with pytest.raises(SystemExit):
        evaluate_baselines.parse_eval_args(
            ["--checkpoint", "rl/models/linear-sarsa.npz"]
        )


def test_checkpoint_eval_keeps_explicit_validation_range():
    arguments = evaluate_baselines.parse_eval_args(
        [
            "--checkpoint",
            "rl/models/linear-sarsa.npz",
            "--start-seed",
            "8000",
            "--episodes",
            "1000",
        ]
    )
    assert arguments.start_seed == 8000
    assert arguments.episodes == 1000


def test_heuristic_eval_defaults_to_final_test():
    arguments = evaluate_baselines.parse_eval_args(["--episodes", "100"])
    assert arguments.start_seed == 10_000
    assert arguments.policy == "all"


def test_evaluate_policy_does_not_update_sarsa_weights():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=4096), SarsaConfig(alpha=0.5), seed=73)
    agent.weights[0, :8] = 1.0
    before = agent.weights.copy()

    result = evaluate_policy(agent, (0,))

    assert result.episodes == 1
    assert agent._frozen
    assert not agent.weights.flags.writeable
    np.testing.assert_array_equal(agent.weights, before)
