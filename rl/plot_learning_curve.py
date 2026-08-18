"""Plot 1,000-episode exploratory training blocks from a Sarsa JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "rl" / "models" / "linear-sarsa-lambda05-8000-full.jsonl"
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "linear-sarsa-learning-curve.pdf"
BLOCK = 1_000


def load_episodes(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        next(handle)
        for line in handle:
            record = json.loads(line)
            if "outcome" in record:
                rows.append(record)
    if not rows:
        raise ValueError(f"No episode records in {path}")
    return rows


def block_stats(rows: list[dict], size: int) -> tuple[list[int], list[float], list[float], list[float]]:
    last_seed = max(row["seed"] for row in rows)
    completed, clear_rate, mean_pellets, mean_return = [], [], [], []
    for start in range(0, last_seed + 1, size):
        chunk = [row for row in rows if start <= row["seed"] < start + size]
        n = len(chunk)
        if n == 0:
            continue
        completed.append(start + size)
        clear_rate.append(sum(row["outcome"] == "cleared" for row in chunk) / n)
        mean_pellets.append(sum(row["pellets"] for row in chunk) / n)
        mean_return.append(sum(row["return"] for row in chunk) / n)
    return completed, clear_rate, mean_pellets, mean_return


def plot(
    completed: list[int],
    clear_rate: list[float],
    mean_pellets: list[float],
    mean_return: list[float],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.2, 5.6), constrained_layout=True)
    series = [
        (axes[0], clear_rate, "Exploratory clear rate", (0.0, 1.0)),
        (axes[1], mean_pellets, "Mean pellets collected", (0.0, 32.0)),
        (axes[2], mean_return, "Mean return", None),
    ]
    for axis, values, ylabel, limits in series:
        axis.plot(completed, values, color="0.15", marker="o", markersize=4.5, linewidth=1.4)
        axis.set_ylabel(ylabel)
        axis.set_xticks(completed)
        axis.grid(True, axis="y", color="0.85", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if limits is not None:
            axis.set_ylim(*limits)
    axes[-1].set_xlabel("Episodes completed (exploratory training, $\\varepsilon > 0$)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block", type=int, default=BLOCK)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.block < 1:
        raise SystemExit("--block must be positive")
    completed, clear_rate, mean_pellets, mean_return = block_stats(
        load_episodes(arguments.log), arguments.block
    )
    plot(completed, clear_rate, mean_pellets, mean_return, arguments.output)
    print(arguments.output)
    print(arguments.output.with_suffix(".png"))
    for done, rate, pellets, ret in zip(completed, clear_rate, mean_pellets, mean_return):
        print(
            f"{done - arguments.block:4d}–{done - 1}: "
            f"clear={rate:.3f} pellets={pellets:.3f} return={ret:.2f}"
        )


if __name__ == "__main__":
    main()
