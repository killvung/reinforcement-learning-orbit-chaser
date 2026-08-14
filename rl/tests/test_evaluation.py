from orbit_chase.evaluation import evaluate_policy
from orbit_chase.policies import RandomValidPolicy


def test_baseline_evaluation_is_reproducible_for_fixed_held_out_seeds():
    seeds = (10_000, 10_001)

    first = evaluate_policy(RandomValidPolicy(), seeds)
    second = evaluate_policy(RandomValidPolicy(), seeds)

    assert first == second
    assert first.episodes == len(seeds)
