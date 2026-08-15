"""Train the documented linear True Online Sarsa(lambda) baseline."""

from __future__ import annotations

import argparse

from orbit_chase.sarsa_training import DEFAULT_LOG_EVERY, TrainingLog, train_linear_sarsa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--agent-seed", type=int, default=73)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print one JSON object per episode to stderr.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Print a cumulative summary to stderr every N episodes (0 disables).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Write config and per-episode JSONL records to this path.",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")

    result = train_linear_sarsa(
        seeds=tuple(range(args.seed_start, args.seed_start + args.episodes)),
        seed=args.agent_seed,
        log=TrainingLog(
            verbose=args.verbose,
            log_every=args.log_every,
            jsonl_path=args.log,
        ),
    )
    print(
        "episodes={episodes} decisions={decisions} clears={clears} captures={captures} "
        "timeouts={timeouts} mean_pellets={pellets:.3f} mean_return={mean_return:.4f} "
        "updated_weights={weights}".format(
            episodes=result.episodes,
            decisions=result.decisions,
            clears=result.clears,
            captures=result.captures,
            timeouts=result.timeouts,
            pellets=result.mean_pellets_collected,
            mean_return=result.mean_return,
            weights=result.updated_weights,
        )
    )


if __name__ == "__main__":
    main()
