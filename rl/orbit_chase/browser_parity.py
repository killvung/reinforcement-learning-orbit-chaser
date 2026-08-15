"""Synthetic linear-Sarsa browser artifact and Python/TS parity cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .checkpoint import browser_artifact_from_agent, save_linear_sarsa
from .environment import OrbitChasePlayerEnv
from .rules import OBSERVATION_SIZE
from .sarsa import FeatureEncoder, LinearSarsaAgent, SarsaConfig

ROOT = Path(__file__).resolve().parents[2]
PARITY_FIXTURE = ROOT / "test" / "fixtures" / "linear-sarsa-parity.json"
SYNTHETIC_CHECKPOINT_NAME = "linear-sarsa-synthetic.npz"


def hand_built_state() -> dict[str, np.ndarray]:
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    observation[0:5] = [0.1, -0.2, 0.3, -0.4, 0.5]
    observation[5:7] = [-0.3, 0.2]
    observation[7] = 1.0
    observation[15] = 0.75
    observation[16:24] = [-0.5, -0.3, -0.1, -0.3, 0.1, 0.2, 0.5, 0.2]
    observation[24:30] = [0.2, 0.1, 1.0, -0.4, 0.2, 1.0]
    observation[120:126] = [0.4, -0.1, 1.0, -0.1, -0.4, 1.0]
    observation[129] = 0.9
    return {
        "observation": observation,
        "action_mask": np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8),
    }


def empty_pellet_state() -> dict[str, np.ndarray]:
    state = hand_built_state()
    state["observation"] = np.array(state["observation"], copy=True)
    state["observation"][24:120] = 0.0
    return state


def seed_reset_state(seed: int = 73) -> dict[str, np.ndarray]:
    environment = OrbitChasePlayerEnv()
    observation, _ = environment.reset(seed=seed)
    return {
        "observation": np.asarray(observation["observation"], dtype=np.float32),
        "action_mask": np.asarray(observation["action_mask"], dtype=np.int8),
    }


def build_synthetic_agent() -> LinearSarsaAgent:
    """Return a tiny agent with distinctive, finite sparse weights."""
    agent = LinearSarsaAgent(FeatureEncoder(capacity=4096), SarsaConfig(alpha=0.05), seed=73)
    state = hand_built_state()
    features = agent.encoder.encode(state)
    agent.weights[0, features.indices[:8]] = np.linspace(0.25, 2.0, 8)
    agent.weights[1, features.indices[4:12]] = np.linspace(-1.5, 1.5, 8)
    agent.weights[2, features.indices[8:16]] = 3.0
    agent.weights[3, features.indices[8:16]] = 3.0
    agent.weights[4, 17] = 0.125
    agent.weights[4, 429] = -0.5
    empty = empty_pellet_state()
    empty_features = agent.encoder.encode(empty)
    agent.weights[5, empty_features.indices[:6]] = np.array([4.0, -2.0, 1.0, 0.5, -0.25, 0.75])
    seeded = seed_reset_state()
    seeded_features = agent.encoder.encode(seeded)
    agent.weights[6, seeded_features.indices[:10]] = np.linspace(-3.0, 2.0, 10)
    agent.weights[7, seeded_features.indices[10:18]] = np.linspace(0.1, 0.8, 8)
    return agent


def _case(name: str, agent: LinearSarsaAgent, state: dict[str, np.ndarray]) -> dict:
    features = agent.encoder.encode(state)
    q_values = agent.q_values(state)
    return {
        "name": name,
        "observation": np.asarray(state["observation"], dtype=np.float64).tolist(),
        "action_mask": np.asarray(state["action_mask"], dtype=np.int8).tolist(),
        "indices": features.indices.astype(int).tolist(),
        "q_values": q_values.tolist(),
        "action": int(agent.choose(state, np.random.default_rng(0))),
    }


def build_parity_document() -> dict:
    agent = build_synthetic_agent()
    return {
        "artifact": browser_artifact_from_agent(agent, SYNTHETIC_CHECKPOINT_NAME),
        "cases": [
            _case("hand-built", agent, hand_built_state()),
            _case("empty-pellets", agent, empty_pellet_state()),
            _case("seed-73-reset", agent, seed_reset_state()),
        ],
    }


def write_parity_fixture(path: Path = PARITY_FIXTURE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_parity_document(), indent=2) + "\n", encoding="utf-8")
    return path


def save_synthetic_checkpoint(directory: str | Path) -> Path:
    return save_linear_sarsa(
        build_synthetic_agent(),
        directory,
        timestamp="synthetic",
        training={"kind": "browser-parity"},
    )
