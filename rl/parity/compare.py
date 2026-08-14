"""Compare TypeScript and Python snapshots for a shared parity fixture."""

from __future__ import annotations

import json
import subprocess
from math import isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compare(left, right, path: str = "root") -> None:
    """Recursively compare JSON values, allowing tiny float differences."""
    if isinstance(left, float) and isinstance(right, float):
        assert isclose(left, right, abs_tol=1e-6), f"{path}: {left} != {right}"
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), f"{path}: different lengths"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare(left_item, right_item, f"{path}[{index}]")
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys(), f"{path}: different keys"
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


def main() -> None:
    typescript = subprocess.run(
        ["node", "test/parity-snapshot.mjs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    python = subprocess.run(
        ["python3", "rl/parity/python_snapshot.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "rl"},
    )
    compare(json.loads(typescript.stdout), json.loads(python.stdout))
    print("TypeScript/Python parity fixture passed.")


if __name__ == "__main__":
    main()
