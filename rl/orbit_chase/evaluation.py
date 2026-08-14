"""Held-out evaluation of fixed player policies."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .rules import (
    DEFAULT_EVALUATION_EPISODES,
    HELD_OUT_SEED_START,
    ROUND_DURATION_SECONDS,
    TerminalOutcome,
)
from .environment import OrbitChasePlayerEnv
from .observation import decode
from .policies import PlayerPolicy


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate held-out episode outcomes for one fixed policy."""

    policy: str
    episodes: int
    clears: int
    captures: int
    timeouts: int
    mean_pellets_collected: float
    mean_return: float
    mean_time_to_clear_seconds: float | None

    @property
    def clear_rate(self) -> float:
        return self.clears / self.episodes

    @property
    def capture_rate(self) -> float:
        return self.captures / self.episodes

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.episodes

    def as_dict(self) -> dict:
        """Return metrics in a stable, JSON-ready form."""
        result = asdict(self)
        result.update(
            clear_rate=self.clear_rate,
            capture_rate=self.capture_rate,
            timeout_rate=self.timeout_rate,
        )
        return result


def evaluate_policy(
    policy: PlayerPolicy,
    seeds: Sequence[int],
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate one fixed policy on explicit held-out episode seeds."""
    if not seeds:
        raise ValueError("Evaluation requires at least one held-out seed.")
    if verbose:
        print(
            f"Evaluating {policy.name} on {len(seeds)} held-out seeds",
            file=sys.stderr,
            flush=True,
        )

    episodes = []
    for index, seed in enumerate(seeds, start=1):
        episode = _play_episode(policy, seed)
        episodes.append(episode)
        if verbose:
            _log_episode(policy.name, index, len(seeds), episode)
    return _summarize_episodes(policy.name, episodes)


def held_out_seeds(
    episodes: int = DEFAULT_EVALUATION_EPISODES,
    start: int = HELD_OUT_SEED_START,
) -> tuple[int, ...]:
    """Return the fixed seed range reserved for baseline evaluation."""
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive.")
    return tuple(range(start, start + episodes))


@dataclass(frozen=True)
class _Episode:
    """One held-out episode: return, pellets, and terminal outcome."""

    seed: int
    episode_return: float
    pellets: int
    outcome: TerminalOutcome
    clear_time_seconds: float | None


def _play_episode(policy: PlayerPolicy, seed: int) -> _Episode:
    """Run one seeded episode to a terminal Gym transition."""
    environment = OrbitChasePlayerEnv()
    state, _ = environment.reset(seed=seed)
    rng = np.random.default_rng(seed)
    episode_return = 0.0
    episode_pellets = 0

    while True:
        action = policy.choose(state, rng)
        state, reward, terminated, truncated, info = environment.step(action)
        episode_return += reward
        episode_pellets += info["pellets_collected"]
        if terminated or truncated:
            outcome = _terminal_outcome(info)
            return _Episode(
                seed=seed,
                episode_return=episode_return,
                pellets=episode_pellets,
                outcome=outcome,
                clear_time_seconds=_clear_time_seconds(outcome, state),
            )


def _terminal_outcome(info: dict[str, bool | int]) -> TerminalOutcome:
    """Pick the first true flag; enum order is capture, then clear, then timeout."""
    return next(outcome for outcome in TerminalOutcome if info[outcome])


def _clear_time_seconds(
    outcome: TerminalOutcome, state: dict[str, np.ndarray]
) -> float | None:
    if outcome is not TerminalOutcome.CLEARED:
        return None
    return (1.0 - decode(state["observation"]).time_fraction) * ROUND_DURATION_SECONDS


def _log_episode(policy_name: str, index: int, total: int, episode: _Episode) -> None:
    print(
        f"{policy_name}  {index}/{total}  seed={episode.seed}  {episode.outcome.name.lower()}  "
        f"pellets={episode.pellets}  return={episode.episode_return:.4f}",
        file=sys.stderr,
        flush=True,
    )


def _summarize_episodes(policy_name: str, episodes: Sequence[_Episode]) -> EvaluationResult:
    clear_times = [
        episode.clear_time_seconds
        for episode in episodes
        if episode.clear_time_seconds is not None
    ]
    return EvaluationResult(
        policy=policy_name,
        episodes=len(episodes),
        clears=sum(episode.outcome is TerminalOutcome.CLEARED for episode in episodes),
        captures=sum(episode.outcome is TerminalOutcome.CAPTURED for episode in episodes),
        timeouts=sum(episode.outcome is TerminalOutcome.TIMEOUT for episode in episodes),
        mean_pellets_collected=round(float(np.mean([episode.pellets for episode in episodes])), 6),
        mean_return=round(float(np.mean([episode.episode_return for episode in episodes])), 6),
        mean_time_to_clear_seconds=(
            round(float(np.mean(clear_times)), 6) if clear_times else None
        ),
    )
