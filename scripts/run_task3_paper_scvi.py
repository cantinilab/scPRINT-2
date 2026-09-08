#!/usr/bin/env python3
"""CLI wrapper for the paper-compatible task3 scVI control."""

from __future__ import annotations

import argparse

from scprint2.evaluation.scvi_baselines import run_task3_paper_scvi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-key", default="orig.ident")
    parser.add_argument("--label-key", default="celltype")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_task3_paper_scvi(parse_args())
