#!/usr/bin/env python3
"""Benchmark task3 embeddings with the modern scib-metrics notebook protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from scprint2.evaluation.task3_scib import (
    parse_embedding_spec,
    run_modern_scib_benchmark,
)


def parse_embedding(specification: str) -> tuple[str, Path, str]:
    return parse_embedding_spec(specification)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--embedding", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-key", default="orig.ident")
    parser.add_argument("--label-key", default="cell_type_ontology_term_id")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    run_modern_scib_benchmark(parse_args())


if __name__ == "__main__":
    main()
