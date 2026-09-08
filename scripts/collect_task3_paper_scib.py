#!/usr/bin/env python3
"""Collect task3 paper-script outputs and apply the paper min-max aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scprint2.evaluation.result_collection import (
    aggregate_task3_paper_scores,
    collect_task3_paper_raw,
    load_task3_reference_raw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, raw_reference = load_task3_reference_raw(args.reference)
    raw = pd.concat(
        [
            raw_reference,
            pd.DataFrame([collect_task3_paper_raw(args.paper_root, args.method)]),
        ],
        ignore_index=True,
    )
    raw.to_csv(args.paper_root / "task3_raw_scores_with_reference.csv", index=False)
    aggregate_task3_paper_scores(raw).to_csv(
        args.paper_root / "task3_summary_scores.csv", index=False
    )


if __name__ == "__main__":
    main()
