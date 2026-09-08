#!/usr/bin/env python3
"""Aggregate several task3 embeddings against one shared paper reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from scprint2.evaluation.result_collection import (
    aggregate_task3_paper_scores,
    aggregate_task3_raw_common_scores,
    build_task3_suite_raw,
    load_task3_reference_raw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="METHOD=RESULT_ROOT",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--include-normalized-diagnostics",
        action="store_true",
        help="Also write the paper-style min-max aggregates for diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, raw = build_task3_suite_raw(args.reference, args.result)
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_root / "task3_suite_raw_scores.csv", index=False)
    aggregate_task3_raw_common_scores(raw).to_csv(
        args.output_root / "task3_suite_raw_common_scores.csv", index=False
    )
    if args.include_normalized_diagnostics:
        aggregate_task3_paper_scores(raw).to_csv(
            args.output_root / "task3_suite_summary_scores.csv", index=False
        )
        _, reference_raw = load_task3_reference_raw(args.reference)
        aggregate_task3_paper_scores(
            raw,
            anchors=reference_raw,
        ).to_csv(
            args.output_root / "task3_suite_paper_anchored_scores.csv", index=False
        )


if __name__ == "__main__":
    main()
