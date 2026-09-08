"""Regenerate OpenProblems figures from snapshots of the manuscript Google Sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

from scprint2.plotting.openproblems import generate_openproblems_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "plot_results_inputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).parent / "plot_results_outputs",
    )
    args = parser.parse_args()

    plot_count = generate_openproblems_plots(args.input_dir, args.output_root)
    print(f"Generated {plot_count} PNG plots under {args.output_root}")


if __name__ == "__main__":
    main()
