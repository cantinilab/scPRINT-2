#!/usr/bin/env python3
"""Run only the missing paper Task3 batch NMI and trajectory metrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_root(paper_root: Path) -> Path:
    root = paper_root / "script"
    nested = root / "benchmark_project_script"
    return nested if nested.is_dir() else root


def run_paper_nmi(root: Path, evaluation: Path) -> Path:
    target = evaluation / "task3_saturn_NMI.csv"
    if target.is_file() and target.stat().st_size:
        return target

    evaluator = evaluator_root(root)
    sys.path.insert(0, str(evaluator / "core_script"))
    from evaluation_ASW_NMI import createAnnData, nmi

    method_dir = root / "output" / "method_outputs" / "saturn"
    adata = createAnnData(
        str(method_dir), "task3_saturn_embedding.txt", ","
    )
    adata.obs["batch"] = adata.obs["batch"].astype(str)
    scores = {
        "nmi_value_celltype": nmi(adata, "cell_type"),
        "nmi_value_batch": nmi(adata, "batch"),
    }
    pd.DataFrame([scores]).to_csv(target)
    return target


def run_paper_trajectory(root: Path, evaluation: Path) -> Path:
    target = evaluation / "task3_saturn_scib_output.csv"
    if target.is_file() and target.stat().st_size:
        return target
    evaluator = evaluator_root(root)
    subprocess.run(
        [
            sys.executable,
            str(evaluator / "scib_metric_running.py"),
            "--method",
            "saturn",
            "--target",
            "task3",
        ],
        cwd=evaluator,
        check=True,
    )
    if not target.is_file() or not target.stat().st_size:
        raise RuntimeError("Deposited scIB evaluator did not write trajectory output")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    root = args.paper_root.resolve()
    output = root / "task3_missing_metrics.csv"
    complete = root / "task3_missing_metrics.COMPLETE"
    metadata_path = root / "task3_missing_metrics.metadata.json"
    if output.exists() or complete.exists():
        raise FileExistsError(f"Refusing to overwrite completed metrics: {output}")

    versions = {
        package: importlib.metadata.version(package)
        for package in ("scib", "scanpy", "anndata", "numpy", "pandas")
    }
    if versions["scib"] != "1.1.3" or versions["scanpy"] != "1.9.1":
        raise RuntimeError(f"Unexpected paper environment: {versions}")

    run_metadata_path = root / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text())
    if run_metadata.get("label_key") != "cell_type_ontology_term_id":
        raise RuntimeError(
            "Task3 missing metrics require cell_type_ontology_term_id labels"
        )

    evaluation = root / "output" / "evaluation" / "saturn"
    ari_path = evaluation / "task3_saturn_ARI.txt"
    if not ari_path.is_file() or not ari_path.stat().st_size:
        raise FileNotFoundError(f"Missing completed paper ARI output: {ari_path}")
    nmi_path = run_paper_nmi(root, evaluation)
    scib_path = run_paper_trajectory(root, evaluation)

    ari = pd.read_table(ari_path).iloc[-1]
    nmi_scores = pd.read_csv(nmi_path).iloc[0]
    scib_scores = pd.read_csv(scib_path, index_col=0).iloc[:, 0]
    result = pd.DataFrame(
        [
            {
                "Method": args.method,
                "Batch.ARI": 1 - max(float(ari["ari_batch"]), 0),
                "Batch.NMI": 1 - float(nmi_scores["nmi_value_batch"]),
                "Trajectory.conservation": float(scib_scores["trajectory"]),
            }
        ]
    )
    if not result.iloc[0, 1:].notna().all():
        raise RuntimeError("A requested Task3 metric is not finite")
    result.to_csv(output, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "protocol": "NAR gkae1316 Figshare 50760384 targeted missing metrics",
                "versions": versions,
                "label_key": run_metadata["label_key"],
                "paper_root": str(root),
                "source_metadata": str(run_metadata_path),
                "input_sha256": {
                    "run_metadata": sha256(run_metadata_path),
                    "ari": sha256(ari_path),
                    "nmi": sha256(nmi_path),
                    "scib": sha256(scib_path),
                },
                "formulas": {
                    "Batch.ARI": "1 - max(ari_batch, 0)",
                    "Batch.NMI": "1 - nmi_value_batch",
                    "Trajectory.conservation": "scib trajectory",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    complete.write_text("COMPLETE\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
