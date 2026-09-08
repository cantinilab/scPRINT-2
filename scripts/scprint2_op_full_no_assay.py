#!/usr/bin/env python3
"""Run scPRINT-2 small-v2 OpenProblems evaluation with raw token embeddings."""

from __future__ import annotations

import argparse

from scprint2.benchmark.openproblems import DATASETS, run_openproblems_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--mode", required=True, choices=("zeroshot", "finetune"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--embedding-view",
        choices=("full_no_assay", "cell_type"),
        default="full_no_assay",
        help=(
            "Embedding scored in zero-shot mode. Fine-tuning always scores the "
            "fine-tuned cell-type token."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mmd-mode",
        choices=("fixed", "off"),
        default="fixed",
        help=(
            "'fixed' enables differentiable pairwise MMD across donor_id groups "
            "on the cell-type token; 'off' disables that regularizer."
        ),
    )
    parser.add_argument(
        "--mmd-weight",
        type=float,
        default=0.03,
        help="Coefficient applied to the attached MMD when --mmd-mode=fixed.",
    )
    parser.add_argument(
        "--mmd-batch-key",
        default="donor_id",
        help=(
            "adata.obs column whose values define MMD groups; for example "
            "donor_id, species, or another batch annotation."
        ),
    )
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--scib-only", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    run_openproblems_benchmark(args)


if __name__ == "__main__":
    main(parse_args())
