"""Export a linear Sarsa `.npz` checkpoint to a sparse browser JSON artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from orbit_chase.checkpoint import (
    DEFAULT_BROWSER_ARTIFACT_DIR,
    export_linear_sarsa_browser,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a linear-sarsa-v1 .npz checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination JSON path. Defaults to "
            "public/models/<checkpoint-stem>.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_args(argv)
    checkpoint = Path(arguments.checkpoint)
    destination = (
        Path(arguments.output)
        if arguments.output is not None
        else DEFAULT_BROWSER_ARTIFACT_DIR / f"{checkpoint.stem}.json"
    )
    path = export_linear_sarsa_browser(checkpoint, destination)
    print(path)


if __name__ == "__main__":
    main()
