import io
import json
import tempfile
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path

from orbit_chase.rules import TerminalOutcome
from orbit_chase.sarsa import FeatureEncoder, SarsaConfig
from orbit_chase.sarsa_training import (
    TrainingLog,
    _terminal_outcome,
    format_rolling_summary,
    train_linear_sarsa,
)


def test_short_seeded_training_run_updates_weights_and_reports_outcomes():
    result = train_linear_sarsa(
        seeds=(0, 1),
        config=SarsaConfig(alpha=0.05, epsilon=0.2),
        encoder=FeatureEncoder(capacity=4096),
        seed=73,
        log=TrainingLog(log_every=0),
    )

    assert result.episodes == 2
    assert result.captures + result.clears + result.timeouts == 2
    assert result.decisions > 0
    assert result.updated_weights > 0
    assert result.agent.traces == {}
    assert result.agent.q_old == 0.0


def test_training_outcome_counts_are_exclusive():
    info = {"captured": True, "cleared": True, "timed_out": False}
    assert _terminal_outcome(info) is TerminalOutcome.CAPTURED
    assert _terminal_outcome({"captured": False, "cleared": True, "timed_out": True}) is (
        TerminalOutcome.CLEARED
    )


def test_verbose_episode_log_is_json():
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        train_linear_sarsa(
            seeds=(0,),
            config=SarsaConfig(alpha=0.05, epsilon=0.2),
            encoder=FeatureEncoder(capacity=4096),
            seed=73,
            log=TrainingLog(verbose=True, log_every=0),
        )

    payload = json.loads(stderr.getvalue().strip())
    assert payload["seed"] == 0
    assert payload["outcome"] in {"captured", "cleared", "timeout"}
    assert payload["trace_count"] > 0
    assert payload["nonzero_weights"] > 0


def test_jsonl_log_writes_config_then_episode():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        train_linear_sarsa(
            seeds=(0,),
            config=SarsaConfig(alpha=0.05, epsilon=0.2),
            encoder=FeatureEncoder(capacity=4096),
            seed=73,
            log=TrainingLog(log_every=0, jsonl_path=path),
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        config = json.loads(lines[0])
        episode = json.loads(lines[1])
        assert config["agent"] == "linear-sarsa"
        assert config["episodes"] == 1
        assert episode["seed"] == 0


def test_after_episode_hook_runs_once_per_episode():
    seen: list[int] = []

    def after_episode(progress):
        seen.append(progress.episode_number)
        assert progress.episode_count == 2
        assert progress.decisions > 0

    train_linear_sarsa(
        seeds=(0, 1),
        config=SarsaConfig(alpha=0.05, epsilon=0.2),
        encoder=FeatureEncoder(capacity=4096),
        seed=73,
        log=TrainingLog(log_every=0, after_episode=after_episode),
    )
    assert seen == [1, 2]


def test_rolling_summary_is_aligned_and_timestamped():
    line = format_rolling_summary(
        episode_number=100,
        episode_count=8000,
        clears=0,
        captures=99,
        timeouts=1,
        returns=[-88.3783],
        pellets=[2],
        abs_deltas=[8.0139],
        now=datetime(2026, 8, 14, 23, 28, 41),
    )
    assert line == (
        "2026-08-14 23:28:41   100/8000  "
        "clear 0.00  capture 0.99  timeout 0.01  "
        "pellets   2.00  return  -88.3783  abs_delta  8.0139"
    )
