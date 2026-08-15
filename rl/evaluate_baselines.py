"""Run reproducible held-out evaluations for player policies and Sarsa checkpoints."""

from __future__ import annotations

import argparse
import json
import sys

from orbit_chase.checkpoint import (
    DEFAULT_ARTIFACT_DIR,
    load_linear_sarsa,
    local_timestamp,
    write_json_artifact,
)
from orbit_chase.evaluation import evaluate_policy, held_out_seeds
from orbit_chase.policies import POLICIES
from orbit_chase.rules import DEFAULT_EVALUATION_EPISODES, HELD_OUT_SEED_START


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVALUATION_EPISODES)
    parser.add_argument("--start-seed", type=int, default=HELD_OUT_SEED_START)
    parser.add_argument("--policy", choices=[*POLICIES, "all"], default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Evaluate a saved linear-sarsa .npz instead of, or as well as, heuristics.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Directory for the timestamped evaluation JSON.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each episode outcome to stderr while evaluation runs.",
    )
    arguments = parser.parse_args()
    if arguments.checkpoint is None and arguments.policy is None:
        arguments.policy = "all"

    seeds = held_out_seeds(arguments.episodes, arguments.start_seed)
    stamp = local_timestamp()
    document: dict = {
        "timestamp": stamp,
        "start_seed": arguments.start_seed,
        "episodes": arguments.episodes,
    }
    names = _heuristic_names(arguments.policy, arguments.checkpoint)
    if names:
        document["results"] = [
            evaluate_policy(POLICIES[name](), seeds, verbose=arguments.verbose).as_dict()
            for name in names
        ]
        path = write_json_artifact(
            "eval-baselines",
            {key: document[key] for key in ("timestamp", "start_seed", "episodes", "results")},
            arguments.artifact_dir,
            timestamp=stamp,
        )
        print(path, file=sys.stderr)
    if arguments.checkpoint is not None:
        sarsa_document = {
            "timestamp": stamp,
            "start_seed": arguments.start_seed,
            "episodes": arguments.episodes,
            "checkpoint": arguments.checkpoint,
            **evaluate_policy(
                load_linear_sarsa(arguments.checkpoint),
                seeds,
                verbose=arguments.verbose,
            ).as_dict(),
        }
        path = write_json_artifact(
            "eval-linear-sarsa",
            sarsa_document,
            arguments.artifact_dir,
            timestamp=stamp,
        )
        print(path, file=sys.stderr)
        document["linear_sarsa"] = sarsa_document
    print(json.dumps(document, indent=2, sort_keys=True))


def _heuristic_names(policy: str | None, checkpoint: str | None) -> list[str]:
    if checkpoint is not None and policy is None:
        return []
    if policy in (None, "all"):
        return list(POLICIES)
    return [policy]


if __name__ == "__main__":
    main()
