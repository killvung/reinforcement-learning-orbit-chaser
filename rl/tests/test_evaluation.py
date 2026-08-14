from orbit_chase.evaluation import _clear_time_seconds, evaluate_policy
from orbit_chase.policies import RandomValidPolicy
from orbit_chase.rules import ROUND_DURATION_SECONDS, TerminalOutcome


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
