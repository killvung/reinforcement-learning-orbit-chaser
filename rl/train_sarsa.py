"""Train the documented linear True Online Sarsa(lambda) baseline."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from orbit_chase.checkpoint import DEFAULT_ARTIFACT_DIR, local_timestamp, save_linear_sarsa
from orbit_chase.sarsa import LinearSarsaAgent, SarsaConfig
from orbit_chase.sarsa_training import (
    DEFAULT_LOG_EVERY,
    EpisodeProgress,
    TrainingLog,
    TrainingResult,
    train_linear_sarsa,
)


def main() -> None:
    defaults = SarsaConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--agent-seed", type=int, default=73)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=defaults.lambda_,
        help="Eligibility trace decay (default: 0.90).",
    )
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
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Directory for timestamped weight checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Write weights every N episodes (0 disables mid-run saves).",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    if not 0.0 <= args.lambda_ <= 1.0:
        parser.error("--lambda must lie in [0, 1]")

    seed_start = args.seed_start
    seeds = tuple(range(seed_start, seed_start + args.episodes))
    run_id = local_timestamp()
    config = SarsaConfig(alpha=args.alpha, lambda_=args.lambda_)

    def after_episode(progress: EpisodeProgress) -> None:
        if args.checkpoint_every <= 0:
            return
        if progress.episode_number % args.checkpoint_every != 0:
            return
        path = _save_progress_checkpoint(
            progress, args.checkpoint_dir, run_id, args.agent_seed, seed_start
        )
        print(f"checkpoint {path}", file=sys.stderr, flush=True)

    result = train_linear_sarsa(
        seeds=seeds,
        config=config,
        seed=args.agent_seed,
        log=TrainingLog(
            verbose=args.verbose,
            log_every=args.log_every,
            jsonl_path=args.log,
            after_episode=after_episode,
        ),
    )
    if not isinstance(result.agent, LinearSarsaAgent):
        raise TypeError("train_linear_sarsa must return a LinearSarsaAgent.")
    checkpoint = save_linear_sarsa(
        result.agent,
        args.checkpoint_dir,
        timestamp=run_id,
        training=_result_training_meta(result, args.agent_seed, seed_start, seeds[-1]),
    )
    print(
        "episodes={episodes} decisions={decisions} clears={clears} captures={captures} "
        "timeouts={timeouts} mean_pellets={pellets:.3f} mean_return={mean_return:.4f} "
        "updated_weights={weights} checkpoint={checkpoint}".format(
            episodes=result.episodes,
            decisions=result.decisions,
            clears=result.clears,
            captures=result.captures,
            timeouts=result.timeouts,
            pellets=result.mean_pellets_collected,
            mean_return=result.mean_return,
            weights=result.updated_weights,
            checkpoint=checkpoint,
        )
    )


def _save_progress_checkpoint(
    progress: EpisodeProgress,
    directory: str,
    run_id: str,
    agent_seed: int,
    seed_start: int,
) -> Path:
    if not isinstance(progress.agent, LinearSarsaAgent):
        raise TypeError("Periodic checkpoints currently support LinearSarsaAgent only.")
    seed_end = seed_start + progress.episode_number - 1
    return save_linear_sarsa(
        progress.agent,
        directory,
        timestamp=f"{run_id}-ep{progress.episode_number:04d}",
        training={
            "agent_seed": agent_seed,
            "seed_start": seed_start,
            "seed_end": seed_end,
            "episodes": progress.episode_number,
            "decisions": progress.decisions,
            "clears": progress.clears,
            "captures": progress.captures,
            "timeouts": progress.timeouts,
            "mean_return": progress.mean_return,
            "mean_pellets_collected": progress.mean_pellets,
            "updated_weights": int(progress.agent.snapshot().get("nonzero_weights", 0)),
        },
    )


def _result_training_meta(
    result: TrainingResult, agent_seed: int, seed_start: int, seed_end: int
) -> dict:
    return {
        "agent_seed": agent_seed,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "episodes": result.episodes,
        "decisions": result.decisions,
        "clears": result.clears,
        "captures": result.captures,
        "timeouts": result.timeouts,
        "mean_return": result.mean_return,
        "mean_pellets_collected": result.mean_pellets_collected,
        "updated_weights": result.updated_weights,
    }


if __name__ == "__main__":
    main()
