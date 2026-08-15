import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from orbit_chase.checkpoint import (
    BROWSER_ACTION_ORDER,
    BROWSER_LINEAR_SARSA_FORMAT,
    LINEAR_SARSA_FORMAT,
    browser_artifact_from_agent,
    export_linear_sarsa_browser,
    load_linear_sarsa,
    save_linear_sarsa,
)
from orbit_chase.rules import ACTION_COUNT, OBSERVATION_SIZE
from orbit_chase.browser_parity import (
    PARITY_FIXTURE,
    SYNTHETIC_CHECKPOINT_NAME,
    build_parity_document,
    build_synthetic_agent,
    write_parity_fixture,
)
from orbit_chase.sarsa import FeatureEncoder, LinearSarsaAgent


def test_browser_export_round_trips_sparse_schema_and_values():
    agent = build_synthetic_agent()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = save_linear_sarsa(agent, directory, timestamp="export-roundtrip")
        destination = Path(directory) / "player.json"
        path = export_linear_sarsa_browser(checkpoint, destination)
        payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["format"] == BROWSER_LINEAR_SARSA_FORMAT
    assert payload["source_format"] == LINEAR_SARSA_FORMAT
    assert payload["checkpoint"] == checkpoint.name
    assert payload["observation_size"] == OBSERVATION_SIZE
    assert payload["action_order"] == list(BROWSER_ACTION_ORDER)
    assert payload["encoder"] == {
        "capacity": agent.encoder.capacity,
        "bins": agent.encoder.bins,
        "tilings": agent.encoder.tilings,
    }
    assert len(payload["weights"]) == ACTION_COUNT
    for action, row in enumerate(payload["weights"]):
        indices = row["indices"]
        values = row["values"]
        assert indices == sorted(indices)
        assert len(indices) == len(values)
        assert np.allclose(values, agent.weights[action, indices])
        assert np.count_nonzero(agent.weights[action]) == len(indices)


def test_browser_export_rejects_unknown_checkpoint_format():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=32), seed=73)
    with tempfile.TemporaryDirectory() as directory:
        path = save_linear_sarsa(agent, directory, timestamp="bad-format")
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
        with pytest.raises(ValueError, match="Unsupported checkpoint format"):
            export_linear_sarsa_browser(path, Path(directory) / "out.json")


def test_browser_export_rejects_nonfinite_weights():
    agent = build_synthetic_agent()
    agent.weights[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        browser_artifact_from_agent(agent, SYNTHETIC_CHECKPOINT_NAME)

    agent = build_synthetic_agent()
    agent.weights[1, 5] = np.inf
    with tempfile.TemporaryDirectory() as directory:
        path = save_linear_sarsa(agent, directory, timestamp="nonfinite")
        with pytest.raises(ValueError, match="finite"):
            export_linear_sarsa_browser(path, Path(directory) / "out.json")


def test_browser_export_rejects_weight_shape_mismatch():
    agent = LinearSarsaAgent(FeatureEncoder(capacity=32), seed=73)
    with tempfile.TemporaryDirectory() as directory:
        path = save_linear_sarsa(agent, directory, timestamp="shape")
        archive = np.load(path)
        meta = json.loads(archive["meta"].tobytes().decode("utf-8"))
        archive.close()
        np.savez_compressed(
            path,
            weights=np.zeros((ACTION_COUNT, 16), dtype=np.float64),
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        )
        with pytest.raises(ValueError, match="weight shape"):
            load_linear_sarsa(path)


def test_committed_parity_fixture_matches_live_encoder_and_choose():
    document = build_parity_document()
    if not PARITY_FIXTURE.exists():
        write_parity_fixture()
    committed = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
    assert committed["artifact"] == document["artifact"]
    assert len(committed["cases"]) == len(document["cases"])
    for expected, actual in zip(committed["cases"], document["cases"]):
        assert expected["name"] == actual["name"]
        assert expected["indices"] == actual["indices"]
        assert expected["action"] == actual["action"]
        assert expected["action_mask"] == actual["action_mask"]
        np.testing.assert_allclose(expected["observation"], actual["observation"], atol=0.0)
        np.testing.assert_allclose(expected["q_values"], actual["q_values"], atol=1e-12)
