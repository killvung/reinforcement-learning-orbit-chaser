import json
import tempfile
from pathlib import Path

import numpy as np

from orbit_chase.checkpoint import (
    LINEAR_SARSA_FORMAT,
    load_linear_sarsa,
    save_linear_sarsa,
    write_json_artifact,
)
from orbit_chase.rules import ACTION_COUNT, OBSERVATION_SIZE
from orbit_chase.sarsa import FeatureEncoder, LinearSarsaAgent, SarsaConfig


def _state() -> dict[str, np.ndarray]:
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


def test_save_and_load_preserves_q_values():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=4096), SarsaConfig(alpha=0.05), seed=73)
    agent.weights[:] = agent.rng.normal(size=agent.weights.shape)
    state = _state()
    before = agent.q_values(state)

    with tempfile.TemporaryDirectory() as directory:
        path = save_linear_sarsa(
            agent,
            directory,
            timestamp="20260814-231500",
            training={"episodes": 3, "agent_seed": 73},
        )
        assert path.name == "linear-sarsa-20260814-231500.npz"
        loaded = load_linear_sarsa(path)

    np.testing.assert_array_equal(loaded.q_values(state), before)
    assert loaded.encoder.capacity == 4096
    assert loaded.config.alpha == 0.05
    assert loaded.traces == {}
    assert loaded.q_old == 0.0


def test_load_rejects_unknown_checkpoint_format():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=32), seed=73)
    with tempfile.TemporaryDirectory() as directory:
        path = save_linear_sarsa(agent, directory, timestamp="20260814-231501")
        archive = np.load(path)
        meta = json.loads(archive["meta"].tobytes().decode("utf-8"))
        weights = archive["weights"]
        archive.close()
        meta["format"] = "not-a-real-format"
        np.savez_compressed(
            path,
            weights=weights,
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        )
        try:
            load_linear_sarsa(path)
        except ValueError as error:
            assert "Unsupported checkpoint format" in str(error)
        else:
            raise AssertionError("load_linear_sarsa should reject an unknown format")


def test_write_json_artifact_includes_timestamp_in_name_and_body():
    with tempfile.TemporaryDirectory() as directory:
        path = write_json_artifact(
            "eval-baselines",
            {"episodes": 2, "results": []},
            directory,
            timestamp="20260814-231502",
        )
        assert path.name == "eval-baselines-20260814-231502.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["timestamp"] == "20260814-231502"
        assert payload["episodes"] == 2


def test_saved_meta_records_format_and_training_scalars():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=32), seed=73)
    with tempfile.TemporaryDirectory() as directory:
        path = save_linear_sarsa(
            agent, directory, timestamp="20260814-231503", training={"clears": 0}
        )
        archive = np.load(path)
        meta = json.loads(archive["meta"].tobytes().decode("utf-8"))
        archive.close()
    assert meta["format"] == LINEAR_SARSA_FORMAT
    assert meta["training"]["clears"] == 0
    assert meta["encoder"]["capacity"] == 32
    assert ACTION_COUNT == 8
