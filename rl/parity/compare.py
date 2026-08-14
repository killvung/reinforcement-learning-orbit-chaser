"""Compare TypeScript and Python snapshots for a shared parity fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from math import isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIRECTORY = ROOT / "rl" / "parity" / "fixtures"


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


def snapshot(command: list[str], fixture: Path) -> dict:
    """Run one snapshot executable against a named shared fixture."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "rl"
    try:
        result = subprocess.run(
            [*command, str(fixture)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"Snapshot failed for {fixture.name}:\n{message}") from error
    return json.loads(result.stdout)


def main() -> None:
    fixtures = sorted(FIXTURES_DIRECTORY.glob("*.json"))
    assert fixtures, "No parity fixtures found."
    for fixture in fixtures:
        typescript = snapshot(["node", "test/parity-snapshot.mjs"], fixture)
        python = snapshot([sys.executable, "rl/parity/python_snapshot.py"], fixture)
        compare(typescript, python, fixture.stem)
        print(f"TypeScript/Python parity fixture passed: {fixture.name}")


if __name__ == "__main__":
    main()
