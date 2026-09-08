#!/usr/bin/env python3
"""Validate and collect the fresh task3-matched Pandora scIB scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scprint2.evaluation.result_collection import collect_pandora_task3_matched_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(".COMPLETE").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    table, metadata = collect_pandora_task3_matched_scores(args.score_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    args.output.with_suffix(".COMPLETE").write_text("COMPLETE\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
