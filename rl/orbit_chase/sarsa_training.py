"""Seeded on-policy training loop for Sarsa agents."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

from .agent import Agent, Transition
from .environment import OrbitChasePlayerEnv
from .rules import TerminalOutcome
from .sarsa import FeatureEncoder, LinearSarsaAgent, SarsaConfig


DEFAULT_FINAL_EPSILON = 0.02
DEFAULT_LOG_EVERY = 100
DEFAULT_EPSILON_HORIZON_EPISODES = 8_000


@dataclass(frozen=True)
class TrainingResult:
    """Inspectable aggregate result from one explicit sequence of training seeds."""

    agent: Agent
    episodes: int
    decisions: int
    clears: int
    captures: int
    timeouts: int
    mean_return: float
    mean_pellets_collected: float
    updated_weights: int


@dataclass(frozen=True)
class EpisodeProgress:
    """Cumulative training state after one finished episode."""

    episode_number: int
    episode_count: int
    agent: Agent
    decisions: int
    clears: int
    captures: int
    timeouts: int
    mean_return: float
    mean_pellets: float
    mean_abs_delta: float


@dataclass(frozen=True)
class TrainingLog:
    """Where to write per-episode records and rolling summaries."""

    verbose: bool = False
    log_every: int = DEFAULT_LOG_EVERY
    jsonl_path: str | Path | None = None
    config_record: dict | None = None
    after_episode: Callable[[EpisodeProgress], None] | None = None


def train_linear_sarsa(
    seeds: Sequence[int],
    config: SarsaConfig = SarsaConfig(),
    encoder: FeatureEncoder | None = None,
    seed: int | None = None,
    final_epsilon: float = DEFAULT_FINAL_EPSILON,
    epsilon_horizon_episodes: int = DEFAULT_EPSILON_HORIZON_EPISODES,
    log: TrainingLog | None = None,
) -> TrainingResult:
    """Construct the linear agent and train on explicit episode seeds."""
    if not 0.0 <= final_epsilon <= config.epsilon:
        raise ValueError("Final epsilon must lie in [0, initial epsilon].")
    agent = LinearSarsaAgent(encoder or FeatureEncoder(), config=config, seed=seed)
    seed_list = tuple(int(value) for value in seeds)
    resolved = log or TrainingLog()
    if resolved.config_record is None:
        resolved = replace(
            resolved,
            config_record=_linear_config_record(
                agent,
                config,
                seed_list,
                seed,
                final_epsilon,
                epsilon_horizon_episodes,
            ),
        )
    return train_agent(
        agent,
        seed_list,
        config.epsilon,
        final_epsilon,
        resolved,
        epsilon_horizon_episodes,
    )


def train_agent(
    agent: Agent,
    seeds: Sequence[int],
    initial_epsilon: float,
    final_epsilon: float,
    log: TrainingLog | None = None,
    epsilon_horizon_episodes: int = DEFAULT_EPSILON_HORIZON_EPISODES,
) -> TrainingResult:
    """On-policy Sarsa control loop over an explicit seed list.

    For each decision: act with masked epsilon-greedy, step the Gym env, then
    `update(Transition)`. At a nonterminal step the next action is sampled
    before the update (one-step Sarsa). A neural Expected-Sarsa agent can
    ignore `next_action` and queue n-step returns inside `update`.
    """
    seed_list = tuple(int(value) for value in seeds)
    if not seed_list:
        raise ValueError("Training requires at least one episode seed.")
    if not 0.0 <= final_epsilon <= initial_epsilon:
        raise ValueError("Final epsilon must lie in [0, initial epsilon].")
    if epsilon_horizon_episodes < 1:
        raise ValueError("Epsilon horizon must be a positive episode count.")

    settings = log or TrainingLog()
    returns: list[float] = []
    pellets: list[int] = []
    abs_deltas: list[float] = []
    decisions = clears = captures = timeouts = 0
    environment = OrbitChasePlayerEnv()
    jsonl = _open_jsonl(settings)

    try:
        if jsonl is not None and settings.config_record is not None:
            jsonl.write(json.dumps(settings.config_record, sort_keys=True) + "\n")
            jsonl.flush()

        for episode_index, episode_seed in enumerate(seed_list):
            epsilon = _epsilon_for_episode(
                episode_index, epsilon_horizon_episodes, initial_epsilon, final_epsilon
            )
            record = _run_episode(agent, environment, episode_seed, epsilon)
            returns.append(record["return"])
            pellets.append(record["pellets"])
            abs_deltas.extend(record["abs_deltas"])
            decisions += record["decisions"]
            if record["outcome"] is TerminalOutcome.CLEARED:
                clears += 1
            elif record["outcome"] is TerminalOutcome.CAPTURED:
                captures += 1
            else:
                timeouts += 1
            _emit_episode_logs(
                settings,
                jsonl,
                episode_index,
                len(seed_list),
                record,
                clears=clears,
                captures=captures,
                timeouts=timeouts,
                returns=returns,
                pellets=pellets,
                abs_deltas=abs_deltas,
            )
            if settings.after_episode is not None:
                settings.after_episode(
                    EpisodeProgress(
                        episode_number=episode_index + 1,
                        episode_count=len(seed_list),
                        agent=agent,
                        decisions=decisions,
                        clears=clears,
                        captures=captures,
                        timeouts=timeouts,
                        mean_return=float(np.mean(returns)),
                        mean_pellets=float(np.mean(pellets)),
                        mean_abs_delta=(
                            float(np.mean(abs_deltas)) if abs_deltas else 0.0
                        ),
                    )
                )
    finally:
        if jsonl is not None:
            jsonl.close()

    snapshot = agent.snapshot()
    return TrainingResult(
        agent=agent,
        episodes=len(seed_list),
        decisions=decisions,
        clears=clears,
        captures=captures,
        timeouts=timeouts,
        mean_return=float(np.mean(returns)),
        mean_pellets_collected=float(np.mean(pellets)),
        updated_weights=int(snapshot.get("nonzero_weights", 0)),
    )


def _run_episode(
    agent: Agent, environment: OrbitChasePlayerEnv, seed: int, epsilon: float
) -> dict:
    """Play one seeded episode and apply on-policy Sarsa updates."""
    state, _ = environment.reset(seed=seed)
    agent.reset_episode()
    action = agent.select_action(state, epsilon)
    episode_return = 0.0
    episode_pellets = 0
    episode_decisions = 0
    abs_deltas: list[float] = []

    while True:
        next_state, reward, terminated, truncated, info = environment.step(action)
        terminal = terminated or truncated
        episode_return += reward
        episode_pellets += int(info["pellets_collected"])
        episode_decisions += 1

        if terminal:
            delta = agent.update(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=None,
                    terminal=True,
                )
            )
            if delta is not None:
                abs_deltas.append(abs(delta))
            snapshot = agent.snapshot()
            agent.reset_episode()
            return {
                "seed": seed,
                "epsilon": epsilon,
                "return": episode_return,
                "pellets": episode_pellets,
                "outcome": _terminal_outcome(info),
                "decisions": episode_decisions,
                "abs_deltas": abs_deltas,
                "snapshot": snapshot,
            }

        next_action = agent.select_action(next_state, epsilon)
        delta = agent.update(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                terminal=False,
            )
        )
        if delta is not None:
            abs_deltas.append(abs(delta))
        state, action = next_state, next_action


def _terminal_outcome(info: dict[str, bool | int]) -> TerminalOutcome:
    """Capture precedes clear, which precedes timeout (enum order)."""
    return next(outcome for outcome in TerminalOutcome if info[outcome])


def _epsilon_for_episode(
    episode_index: int,
    horizon_episodes: int,
    initial: float,
    final: float,
) -> float:
    """Linearly decay epsilon across a fixed episode horizon, not the run length."""
    if horizon_episodes <= 1:
        return final
    fraction = min(1.0, episode_index / (horizon_episodes - 1))
    return initial + fraction * (final - initial)


def _linear_config_record(
    agent: LinearSarsaAgent,
    config: SarsaConfig,
    seeds: tuple[int, ...],
    agent_seed: int | None,
    final_epsilon: float,
    epsilon_horizon_episodes: int,
) -> dict:
    return {
        "agent": agent.name,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "lambda": config.lambda_,
        "epsilon_initial": config.epsilon,
        "epsilon_final": final_epsilon,
        "epsilon_horizon_episodes": epsilon_horizon_episodes,
        "feature_capacity": agent.encoder.capacity,
        "tile_bins": agent.encoder.bins,
        "tile_tilings": agent.encoder.tilings,
        "agent_seed": agent_seed,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "episodes": len(seeds),
    }


def _open_jsonl(settings: TrainingLog) -> TextIO | None:
    if settings.jsonl_path is None:
        return None
    return Path(settings.jsonl_path).expanduser().open("w", encoding="utf-8")


def _emit_episode_logs(
    settings: TrainingLog,
    jsonl: TextIO | None,
    episode_index: int,
    episode_count: int,
    record: dict,
    *,
    clears: int,
    captures: int,
    timeouts: int,
    returns: list[float],
    pellets: list[int],
    abs_deltas: list[float],
) -> None:
    payload = _episode_payload(record)
    encoded = json.dumps(payload, sort_keys=True)
    if settings.verbose:
        print(encoded, file=sys.stderr, flush=True)
    if jsonl is not None:
        jsonl.write(encoded + "\n")
        jsonl.flush()
    log_every = settings.log_every
    episode_number = episode_index + 1
    if log_every > 0 and (
        episode_number % log_every == 0 or episode_number == episode_count
    ):
        print(
            format_rolling_summary(
                episode_number,
                episode_count,
                clears,
                captures,
                timeouts,
                returns,
                pellets,
                abs_deltas,
            ),
            file=sys.stderr,
            flush=True,
        )


def _episode_payload(record: dict) -> dict:
    episode_deltas = record["abs_deltas"]
    payload = {
        "seed": record["seed"],
        "epsilon": record["epsilon"],
        "return": record["return"],
        "pellets": record["pellets"],
        "outcome": record["outcome"].name.lower(),
        "decisions": record["decisions"],
        "mean_abs_delta": float(np.mean(episode_deltas)) if episode_deltas else 0.0,
    }
    payload.update(record["snapshot"])
    return payload


def format_rolling_summary(
    episode_number: int,
    episode_count: int,
    clears: int,
    captures: int,
    timeouts: int,
    returns: list[float],
    pellets: list[int],
    abs_deltas: list[float],
    now: datetime | None = None,
) -> str:
    """One aligned stderr line for a cumulative training checkpoint."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    width = len(str(episode_count))
    mean_delta = float(np.mean(abs_deltas)) if abs_deltas else 0.0
    return (
        f"{stamp}  {episode_number:>{width}}/{episode_count}  "
        f"clear {clears / episode_number:4.2f}  "
        f"capture {captures / episode_number:4.2f}  "
        f"timeout {timeouts / episode_number:4.2f}  "
        f"pellets {float(np.mean(pellets)):6.2f}  "
        f"return {float(np.mean(returns)):9.4f}  "
        f"abs_delta {mean_delta:7.4f}"
    )
