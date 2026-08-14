"""Run reproducible held-out evaluations for the fixed player baselines."""

from __future__ import annotations

import argparse
import json

from orbit_chase.rules import DEFAULT_EVALUATION_EPISODES, HELD_OUT_SEED_START
from orbit_chase.evaluation import evaluate_policy, held_out_seeds
from orbit_chase.policies import POLICIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVALUATION_EPISODES)
    parser.add_argument("--start-seed", type=int, default=HELD_OUT_SEED_START)
    parser.add_argument("--policy", choices=[*POLICIES, "all"], default="all")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each episode outcome to stderr while evaluation runs.",
    )
    arguments = parser.parse_args()

    seeds = held_out_seeds(arguments.episodes, arguments.start_seed)
    names = list(POLICIES) if arguments.policy == "all" else [arguments.policy]
    results = [
        evaluate_policy(POLICIES[name](), seeds, verbose=arguments.verbose).as_dict()
        for name in names
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
