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

VALIDATION_SEED_START = 8_000
VALIDATION_SEED_END = 8_999
FINAL_TEST_SEED_END = HELD_OUT_SEED_START + DEFAULT_EVALUATION_EPISODES - 1


def seed_split_label(start_seed: int, episodes: int) -> str:
    """Name the documented split covered by `[start_seed, start_seed + episodes)`."""
    end_seed = start_seed + episodes - 1
    if start_seed == VALIDATION_SEED_START and end_seed <= VALIDATION_SEED_END:
        return "validation"
    if start_seed == HELD_OUT_SEED_START and end_seed <= FINAL_TEST_SEED_END:
        return "final-test"
    if 0 <= start_seed and end_seed <= 7_999:
        return "training"
    return "custom"


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse eval flags. `--checkpoint` requires an explicit `--start-seed`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVALUATION_EPISODES)
    parser.add_argument(
        "--start-seed",
        type=int,
        default=None,
        help=(
            "First episode seed. Required with --checkpoint. "
            "8000=validation (through 8999), 10000=final-test (through 10099). "
            "Heuristic-only runs default to 10000."
        ),
    )
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
    arguments = parser.parse_args(argv)
    if arguments.checkpoint is None and arguments.policy is None:
        arguments.policy = "all"
    if arguments.episodes < 1:
        parser.error("--episodes must be positive")
    if arguments.start_seed is None:
        if arguments.checkpoint is not None:
            parser.error(
                "--start-seed is required with --checkpoint "
                "(8000=validation, 10000=final-test)."
            )
        arguments.start_seed = HELD_OUT_SEED_START
    return arguments


def main() -> None:
    arguments = parse_eval_args()
    end_seed = arguments.start_seed + arguments.episodes - 1
    split = seed_split_label(arguments.start_seed, arguments.episodes)
    print(
        f"evaluating seeds {arguments.start_seed}–{end_seed} "
        f"({arguments.episodes} episodes, {split})",
        file=sys.stderr,
        flush=True,
    )

    seeds = held_out_seeds(arguments.episodes, arguments.start_seed)
    stamp = local_timestamp()
    document: dict = {
        "timestamp": stamp,
        "start_seed": arguments.start_seed,
        "seed_end": end_seed,
        "split": split,
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
            {
                key: document[key]
                for key in (
                    "timestamp",
                    "start_seed",
                    "seed_end",
                    "split",
                    "episodes",
                    "results",
                )
            },
            arguments.artifact_dir,
            timestamp=stamp,
        )
        print(path, file=sys.stderr)
    if arguments.checkpoint is not None:
        agent = load_linear_sarsa(arguments.checkpoint)
        agent.freeze()
        sarsa_document = {
            "timestamp": stamp,
            "start_seed": arguments.start_seed,
            "seed_end": end_seed,
            "split": split,
            "episodes": arguments.episodes,
            "checkpoint": arguments.checkpoint,
            **evaluate_policy(
                agent,
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
