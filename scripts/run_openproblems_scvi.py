#!/usr/bin/env python3
"""CLI wrapper for the OpenProblems scVI baseline."""

from __future__ import annotations

import argparse

from scprint2.evaluation.scvi_baselines import run_openproblems_scvi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-latent", type=int, default=30)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run_openproblems_scvi(parse_args())
