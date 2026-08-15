"""Timestamped on-disk artifacts for linear Sarsa and evaluation JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .rules import ACTION_COUNT
from .sarsa import FeatureEncoder, LinearSarsaAgent, SarsaConfig


LINEAR_SARSA_FORMAT = "orbit-chase-linear-sarsa-v1"
DEFAULT_ARTIFACT_DIR = Path("rl/models")


def local_timestamp() -> str:
    """Return a filesystem-safe local timestamp `YYYYMMDD-HHMMSS`."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_linear_sarsa(
    agent: LinearSarsaAgent,
    directory: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    timestamp: str | None = None,
    training: dict[str, Any] | None = None,
) -> Path:
    """Write `linear-sarsa-{timestamp}.npz` with weights and JSON metadata."""
    if not isinstance(agent, LinearSarsaAgent):
        raise TypeError("Checkpoints currently support LinearSarsaAgent only.")
    stamp = timestamp or local_timestamp()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"linear-sarsa-{stamp}.npz"
    meta = {
        "format": LINEAR_SARSA_FORMAT,
        "timestamp": stamp,
        "config": {
            "alpha": agent.config.alpha,
            "gamma": agent.config.gamma,
            "lambda": agent.config.lambda_,
            "epsilon": agent.config.epsilon,
        },
        "encoder": {
            "capacity": agent.encoder.capacity,
            "bins": agent.encoder.bins,
            "tilings": agent.encoder.tilings,
        },
        "training": training or {},
    }
    np.savez_compressed(
        path,
        weights=np.asarray(agent.weights, dtype=np.float64),
        meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8),
    )
    return path


def load_linear_sarsa(path: str | Path) -> LinearSarsaAgent:
    """Rebuild a linear agent from a v1 checkpoint. Traces and q_old stay empty."""
    archive = np.load(Path(path), allow_pickle=False)
    try:
        meta = json.loads(archive["meta"].tobytes().decode("utf-8"))
        weights = np.asarray(archive["weights"], dtype=np.float64)
    finally:
        archive.close()
    if meta.get("format") != LINEAR_SARSA_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {meta.get('format')!r}")
    encoder_meta = meta["encoder"]
    encoder = FeatureEncoder(
        capacity=int(encoder_meta["capacity"]),
        bins=int(encoder_meta["bins"]),
        tilings=int(encoder_meta["tilings"]),
    )
    config_meta = meta["config"]
    config = SarsaConfig(
        alpha=float(config_meta["alpha"]),
        gamma=float(config_meta["gamma"]),
        lambda_=float(config_meta["lambda"]),
        epsilon=float(config_meta["epsilon"]),
    )
    agent = LinearSarsaAgent(encoder, config=config)
    if weights.shape != (ACTION_COUNT, encoder.capacity):
        raise ValueError(
            f"Checkpoint weight shape {weights.shape} does not match "
            f"({ACTION_COUNT}, {encoder.capacity})."
        )
    agent.weights[:, :] = weights
    return agent


def write_json_artifact(
    kind: str,
    payload: dict[str, Any],
    directory: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    timestamp: str | None = None,
) -> Path:
    """Write `{kind}-{timestamp}.json` and return the path."""
    stamp = timestamp or local_timestamp()
    document = dict(payload)
    document.setdefault("timestamp", stamp)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-{stamp}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
