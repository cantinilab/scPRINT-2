#!/usr/bin/env python3
"""Score a preserved embedding with the repository OpenProblems evaluator."""

from __future__ import annotations

import argparse

from scprint2.evaluation.openproblems_scib import (
    score_preserved_op_embedding,
)


def main(args: argparse.Namespace) -> None:
    result = score_preserved_op_embedding(
        input_path=args.input,
        output_path=args.output,
        dataset=args.dataset,
        embedding_key=args.embedding_key,
        batch_key=args.batch_key,
        label_key=args.label_key,
        method_id=args.method_id,
        trust_positional_order=args.trust_positional_order,
    )
    print(result.to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--embedding-key", default="X_emb")
    parser.add_argument("--batch-key", default="donor_id")
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--method-id", default="scvi_openproblems_recipe")
    parser.add_argument(
        "--trust-positional-order",
        action="store_true",
        help=(
            "Align observation names positionally only after batch and label columns "
            "match the OpenProblems solution exactly."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
