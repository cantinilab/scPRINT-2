#!/usr/bin/env python3
"""Check scPRINT-2 classifier-logit to ontology-label mappings on a sample."""

from __future__ import annotations

import argparse
import json

import numpy as np
import scanpy as sc

from scprint2 import scPRINT2
from scprint2.tasks import Embedder
from scprint2.tasks.cell_emb import compute_classification

CELL_TYPE = "cell_type_ontology_term_id"


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred == truth))


def hierarchy_accuracy(
    winners: np.ndarray,
    labels: list[str],
    truth: np.ndarray,
    decoder: dict[int, str],
    hierarchy: dict[int, list[int]],
) -> float:
    reverse = {label: index for index, label in decoder.items()}
    correct = []
    for winner, true in zip(winners, truth, strict=True):
        true_index = reverse.get(true)
        correct.append(
            labels[winner] == true
            or (
                true_index in hierarchy
                and int(winner) in hierarchy[true_index]
            )
        )
    return float(np.mean(correct))


def main(args: argparse.Namespace) -> None:
    source = sc.read_h5ad(args.input, backed="r")
    if source.n_obs > args.cells:
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(source.n_obs, args.cells, replace=False))
        adata = source[rows].to_memory()
    else:
        adata = source.to_memory()
    source.file.close()
    model = scPRINT2.load_from_checkpoint(
        args.checkpoint, precpt_gene_emb=None, gene_pos_file=None
    ).to("cuda")
    model.mask_zeros = args.mask_zeros
    embedded, _ = Embedder(
        how="random expr",
        max_len=args.max_len,
        num_workers=8,
        pred_embedding=[CELL_TYPE],
        keep_all_labels_pred=True,
        doplot=False,
    )(model, adata)
    decoder = model.label_decoders[CELL_TYPE]
    width = model.label_counts[CELL_TYPE]
    indexed = [decoder[i] for i in range(width)]
    columns = list(embedded.obs.columns[embedded.obs.columns.str.startswith("CL:")])
    if columns != indexed:
        raise AssertionError("Embedder CL columns are not decoder[0:n] in order")
    logits = embedded.obs.loc[:, columns].to_numpy(dtype=np.float32)
    winners = logits.argmax(1)
    truth = embedded.obs[CELL_TYPE].astype(str).to_numpy()
    insertion = list(decoder.values())[:width]
    sorted_nonnegative = [decoder[i] for i in sorted(k for k in decoder if k >= 0)[:width]]
    candidates = {
        "decoder_index": indexed,
        "dict_insertion": insertion,
        "sorted_nonnegative_keys": sorted_nonnegative,
    }
    report = {
        "cells": embedded.n_obs,
        "classifier_width": width,
        "decoder_entries": len(decoder),
        "accuracies": {
            name: accuracy(np.asarray(labels)[winners], truth)
            for name, labels in candidates.items()
        },
        "top_truth": embedded.obs[CELL_TYPE].value_counts().head(15).to_dict(),
        "top_predictions_by_mapping": {},
    }
    hierarchy = model.labels_hierarchy.get(CELL_TYPE, {})
    reverse = {label: index for index, label in decoder.items()}
    report["hierarchy_accuracy_decoder_index"] = hierarchy_accuracy(
        winners, indexed, truth, decoder, hierarchy
    )
    report["truth_decoder_coverage"] = {
        "leaf": int(sum(0 <= reverse.get(label, -1) < width for label in truth)),
        "parent": int(sum(reverse.get(label, -1) in hierarchy for label in truth)),
        "unavailable": int(sum(label not in reverse for label in truth)),
    }
    embedded.obs[f"pred_{CELL_TYPE}"] = np.asarray(indexed)[winners]
    report["project_helper"] = compute_classification(
        embedded,
        [CELL_TYPE],
        label_decoders=model.label_decoders,
        labels_hierarchy=model.labels_hierarchy,
    )[CELL_TYPE]
    for name, labels in candidates.items():
        values, counts = np.unique(np.asarray(labels)[winners], return_counts=True)
        order = np.argsort(counts)[::-1][:15]
        report["top_predictions_by_mapping"][name] = {
            str(values[i]): int(counts[i]) for i in order
        }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--max-len", type=int, default=3_200)
    parser.add_argument(
        "--mask-zeros", action=argparse.BooleanOptionalAction, default=False
    )
    main(parser.parse_args())
