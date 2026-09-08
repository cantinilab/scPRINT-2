#!/usr/bin/env python3
"""Collect completed Pandora scIB 1.1.3 scores and compare modern totals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scprint2.evaluation.result_collection import collect_pandora_scib113_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--modern", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy, comparison, metadata = collect_pandora_scib113_scores(
        args.data_root, args.modern
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    legacy.to_csv(args.output, sep="\t", index=False)

    comparison_path = args.output.with_name(args.output.stem + "_vs_modern.tsv")
    comparison.to_csv(comparison_path, sep="\t", index=False)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "legacy_scores": str(args.output),
                "modern_scores": str(args.modern),
                "comparison": str(comparison_path),
                "methods": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    args.output.with_suffix(".COMPLETE").write_text("COMPLETE\n")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
