#!/usr/bin/env python3
"""Evaluate OpenProblems-style label projection on saved scPRINT outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn import set_config

from scprint2.evaluation.label_projection import (
    CELL_TYPE,
    evaluate,
    load_descendants,
    load_saved_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output")
    parser.add_argument("--truth-key", default=CELL_TYPE)
    parser.add_argument("--held-out-key", default="classification_held_out")
    parser.add_argument(
        "--prediction-key",
        default="pred_cell_type_ontology_term_id_direct",
    )
    parser.add_argument(
        "--embedding-key",
        action="append",
        dest="embedding_keys",
        default=None,
    )
    parser.add_argument("--working-memory-mb", type=int, default=4096)
    parser.add_argument(
        "--knn-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--knn-block-size", type=int, default=512)
    parser.add_argument("--skip-mappings", action="store_true")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument(
        "--predictions-output",
        help="Optional compact per-test-cell CSV or CSV.GZ prediction output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    metadata_path = (
        Path(args.metadata) if args.metadata else input_path.with_suffix(".meta.json")
    )
    embedding_keys = args.embedding_keys or (
        []
        if args.skip_knn
        else ["scprint_emb_cell_type_ontology_term_id", "scprint_emb"]
    )
    set_config(working_memory=args.working_memory_mb)
    knn_device = args.knn_device
    if knn_device == "auto":
        knn_device = "cuda" if torch.cuda.is_available() else "cpu"
    if knn_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--knn-device=cuda requested but CUDA is unavailable")
    arrays = load_saved_arrays(
        input_path,
        truth_key=args.truth_key,
        held_out_key=args.held_out_key,
        prediction_key=args.prediction_key,
        embedding_keys=embedding_keys,
    )
    descendants = load_descendants(metadata_path)
    evaluation, predictions = evaluate(
        arrays,
        descendants,
        embedding_keys,
        knn_device=knn_device,
        knn_block_size=args.knn_block_size,
        run_mappings=not args.skip_mappings,
        run_knn=not args.skip_knn,
    )
    report = {
        "input": str(input_path),
        "metadata": str(metadata_path),
        "prediction_key": args.prediction_key,
        "embedding_keys": embedding_keys,
        "knn_device": knn_device,
        **evaluation,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    if args.predictions_output:
        prediction_path = Path(args.predictions_output)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
