#!/usr/bin/env python3
"""Prepare a task3 embedding for the exact cross-species paper scIB scripts."""

from __future__ import annotations

import argparse

from scprint2.evaluation.task3_scib import (
    PAPER_SCANPY_VERSION as _PAPER_SCANPY_VERSION,
)
from scprint2.evaluation.task3_scib import (
    PAPER_SCIB_VERSION as _PAPER_SCIB_VERSION,
)
from scprint2.evaluation.task3_scib import (
    aligned_embedding,
    assert_task3_scib113_environment,
    prepare_paper_scib,
    sha256,
)

_sha256 = sha256
_aligned_embedding = aligned_embedding
PAPER_SCIB_VERSION = _PAPER_SCIB_VERSION
PAPER_SCANPY_VERSION = _PAPER_SCANPY_VERSION


def _assert_paper_environment() -> dict[str, str]:
    return assert_task3_scib113_environment(
        mismatch_message="Paper environment mismatch"
    )


def prepare(args: argparse.Namespace) -> None:
    prepare_paper_scib(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-key", required=True)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--batch-key", default="orig.ident")
    parser.add_argument("--label-key", default="celltype")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
