#!/usr/bin/env python3
"""Compatibility CLI wrapper for the task3 cross-species embedding runner."""

from __future__ import annotations

from scprint2.benchmark.task3 import (
    Task3Config,
    Task3RunArtifacts,
    _align_to_checkpoint_genes,
    _classification_targets,
    _common_metadata,
    _ensure_historical_knn_graph,
    _finetune_settings,
    _json_default,
    _load_scprint,
    _mmd_class_token,
    _refuse_overwrite,
    _unknown_checkpoint_labels,
    _valid_label_mask,
    main,
    run_scprint,
    run_task3,
    run_transcriptformer,
)

__all__ = [
    "Task3Config",
    "Task3RunArtifacts",
    "_align_to_checkpoint_genes",
    "_classification_targets",
    "_common_metadata",
    "_ensure_historical_knn_graph",
    "_finetune_settings",
    "_json_default",
    "_load_scprint",
    "_mmd_class_token",
    "_unknown_checkpoint_labels",
    "_refuse_overwrite",
    "_valid_label_mask",
    "main",
    "run_scprint",
    "run_task3",
    "run_transcriptformer",
]


if __name__ == "__main__":
    main()
