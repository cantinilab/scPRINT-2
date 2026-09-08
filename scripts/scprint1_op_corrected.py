#!/usr/bin/env python3
"""Recompute the four OpenProblems datasets with the corrected scPRINT-1 gene map."""

from __future__ import annotations

import argparse

from scprint2.benchmark.openproblems_v1 import (
    DATASETS,
    run_openproblems_v1_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    run_openproblems_v1_benchmark(args)


if __name__ == "__main__":
    main(parse_args())
