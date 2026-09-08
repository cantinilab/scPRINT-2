#!/usr/bin/env python3
"""Clean, parameterized runners for the three task3 cross-species notebooks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TASK3_MODES = (
    "scprint-zero-shot",
    "scprint1-zero-shot",
    "scprint2-zero-shot",
    "scprint-mmd",
    "scprint2-ft",
    "transcriptformer",
)


@dataclass(frozen=True)
class Task3Config:
    """Python-facing configuration for an auditable Task3 notebook run."""

    mode: str
    input: Path | str
    checkpoint: Path | str
    output: Path | str
    notebook_provenance: str
    batch_key: str = "orig.ident"
    cell_type_key: str = "cell_type_ontology_term_id"
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 8
    embed_how: str = "random expr"
    embed_max_len: int = 2200
    finetune_max_len: int = 2200
    use_knn: bool = False
    rebuild_knn_graph: bool = False
    legacy_detached_mmd: bool = False
    restore_best_model: bool = True
    finetune_objective: str = "reconstruction-mmd"
    mmd_target: str = "species"
    mmd_scale: float = 0.03
    include_organism_classification: bool = False
    exclude_train_label: tuple[str, ...] = ("unknown",)
    include_unknown_train_labels: bool = False
    require_known_train_labels: bool = False
    feature_organism: str | None = None
    seed: int = 42
    align_checkpoint_genes: bool = False

    def __post_init__(self) -> None:
        if self.mode not in TASK3_MODES:
            raise ValueError(f"Unsupported Task3 mode: {self.mode}")
        if self.finetune_objective not in {"reconstruction-mmd", "cell-type-only"}:
            raise ValueError(
                f"Unsupported fine-tuning objective: {self.finetune_objective}"
            )
        if self.mmd_target not in {"label", "species"}:
            raise ValueError(f"Unsupported MMD target label: {self.mmd_target}")
        if self.mmd_scale < 0:
            raise ValueError("mmd_scale must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("input", "checkpoint", "output"):
            values[key] = str(values[key])
        values["exclude_train_label"] = list(values["exclude_train_label"])
        return values

    def as_namespace(self) -> argparse.Namespace:
        values = asdict(self)
        values["exclude_train_label"] = list(values["exclude_train_label"])
        return argparse.Namespace(**values)


@dataclass(frozen=True)
class Task3RunArtifacts:
    output: Path
    metadata: Path
    checkpoint: Path | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "output": str(self.output),
            "metadata": str(self.metadata),
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
        }


def task3_output_paths(config: Task3Config) -> list[Path]:
    """Return every artifact path reserved by a scPRINT Task3 run."""
    output = Path(config.output)
    paths = [output, output.with_suffix(".metadata.json")]
    if config.mode in {"scprint-mmd", "scprint2-ft"}:
        paths.append(output.with_suffix(".ckpt"))
    return paths


def validate_task3_output_paths(config: Task3Config) -> list[Path]:
    """Refuse a partial overwrite before any expensive Task3 work starts."""
    paths = task3_output_paths(config)
    _refuse_overwrite(*paths)
    return paths


def seed_task3(seed: int) -> None:
    """Seed the libraries used by the scPRINT Task3 path."""
    import lightning as L
    import numpy as np
    import torch

    L.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_task3_dataset(config: Task3Config):
    """Load the Task3 expression input without mutating it."""
    import scanpy as sc

    return sc.read_h5ad(config.input)


def summarize_task3_dataset(adata, config: Task3Config):
    """Return the dataset checks that should be visible in a notebook."""
    import pandas as pd

    missing = [
        key
        for key in (config.batch_key, config.cell_type_key)
        if key not in adata.obs
    ]
    if missing:
        raise KeyError(f"Missing Task3 obs columns: {missing}")
    labels = adata.obs[config.cell_type_key].astype(str)
    batches = adata.obs[config.batch_key].astype(str)
    return pd.Series(
        {
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "labels": int(labels.nunique()),
            "batches": int(batches.nunique()),
            "unknown_labels": int(labels.str.casefold().eq("unknown").sum()),
            "batch_key": config.batch_key,
            "label_key": config.cell_type_key,
        },
        name="Task3 dataset audit",
    )


def load_task3_checkpoint(config: Task3Config):
    """Load the scPRINT checkpoint selected by a Task3 configuration."""
    return _load_scprint(str(config.checkpoint))


def prepare_task3_dataset(adata, model, config: Task3Config):
    """Build the optional KNN input graph and align checkpoint genes."""
    rebuilt = False
    if config.use_knn and model.expr_emb_style == "metacell":
        rebuilt = _ensure_historical_knn_graph(
            adata,
            seed=config.seed,
            force_rebuild=config.rebuild_knn_graph,
        )
    if config.align_checkpoint_genes:
        adata = _align_to_checkpoint_genes(adata, model)
    return adata, rebuilt


def finetune_task3_checkpoint(model, adata, config: Task3Config):
    """Fine-tune one checkpoint and return its explicit training provenance."""
    import lightning as L
    import torch

    from scprint2.tasks import FinetuneBatchClass

    if config.mode not in {"scprint-mmd", "scprint2-ft"}:
        return model, None, None
    if config.cell_type_key not in adata.obs:
        raise KeyError(f"Missing obs[{config.cell_type_key!r}] for fine-tuning")
    unknown_checkpoint_labels = _unknown_checkpoint_labels(
        model,
        adata.obs[config.cell_type_key],
        config.cell_type_key,
    )
    if config.require_known_train_labels and unknown_checkpoint_labels:
        raise ValueError(
            "Fine-tuning labels are absent from the checkpoint decoder: "
            + ", ".join(unknown_checkpoint_labels)
        )
    excluded = (
        set()
        if config.include_unknown_train_labels
        else {value.strip().casefold() for value in config.exclude_train_label}
    )
    train_keep = _valid_label_mask(adata.obs[config.cell_type_key], excluded)
    if not train_keep.any():
        raise RuntimeError("No cells remain for fine-tuning after label filtering")
    train_adata = adata[train_keep].copy()
    args = config.as_namespace()
    settings = _finetune_settings(args)
    finetuner = FinetuneBatchClass(
        batch_key=config.batch_key,
        predict_keys=_classification_targets(args),
        max_len=config.finetune_max_len,
        learn_batches_on=settings["learn_batches_on"],
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        do_mmd_on=settings["do_mmd_on"],
        lr=0.0002,
        ft_mode="xpressor",
        frac_train=0.8,
        loss_scalers=settings["loss_scalers"],
        use_knn=config.use_knn,
        legacy_detached_mmd=config.legacy_detached_mmd,
        restore_best_model=config.restore_best_model,
        train_organism_decoder=False,
    )
    model = finetuner(model, adata=train_adata)
    training_filter = {
        "label_key": config.cell_type_key,
        "excluded_labels": sorted(excluded),
        "source_cells": int(adata.n_obs),
        "training_cells": int(train_adata.n_obs),
        "excluded_cells": int(adata.n_obs - train_adata.n_obs),
        "embedding_cells_after_finetune": int(adata.n_obs),
        "checkpoint_unknown_labels": unknown_checkpoint_labels,
        "require_known_train_labels": bool(config.require_known_train_labels),
        "mmd_target": config.mmd_target,
        "mmd_batch_key": config.batch_key,
        "finetune_objective": config.finetune_objective,
        "do_mmd_on": settings["do_mmd_on"],
        "classification_targets": _classification_targets(args),
        "organism_classification": bool(config.include_organism_classification),
        "knn_forward_used": bool(
            config.use_knn and model.expr_emb_style == "metacell"
        ),
        "best_validation_epoch": finetuner.best_epoch,
        "best_validation_loss": finetuner.best_val_loss,
    }
    checkpoint_output = Path(config.output).with_suffix(".ckpt")
    torch.save(
        {
            "epoch": (
                finetuner.best_epoch - 1
                if config.restore_best_model and finetuner.best_epoch is not None
                else config.num_epochs - 1
            ),
            "global_step": 0,
            "pytorch-lightning_version": L.__version__,
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
        },
        checkpoint_output,
    )
    return model, training_filter, checkpoint_output


def embed_task3(model, adata, config: Task3Config):
    """Compute the configured Task3 embedding without saving it."""
    from scprint2.tasks import Embedder

    embedded, _ = Embedder(
        how=config.embed_how,
        max_len=config.embed_max_len,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pred_embedding=["all"],
        doplot=False,
        save_every=10_000,
        use_knn=config.use_knn,
    )(model, adata)
    if "scprint_emb" not in embedded.obsm:
        raise RuntimeError("scPRINT embedding completed without obsm['scprint_emb']")
    return embedded


def save_task3_embedding(
    embedded,
    prepared_adata,
    config: Task3Config,
    *,
    training_filter: dict | None = None,
    knn_graph_rebuilt: bool = False,
) -> Task3RunArtifacts:
    """Persist the embedding and its complete Task3 provenance."""
    output = Path(config.output)
    embedded.write_h5ad(output, compression="lzf")
    args = config.as_namespace()
    args.knn_graph_rebuilt = knn_graph_rebuilt
    metadata_path = _common_metadata(
        args,
        "scprint_emb",
        adata=prepared_adata,
        training_filter=training_filter,
    )
    checkpoint_path = output.with_suffix(".ckpt")
    return Task3RunArtifacts(
        output=output,
        metadata=metadata_path,
        checkpoint=checkpoint_path if checkpoint_path.exists() else None,
    )


def _json_default(value):
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _refuse_overwrite(*paths: Path) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s): "
            + ", ".join(map(str, existing))
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _valid_label_mask(labels, excluded: set[str]):
    normalized = labels.astype(str).str.strip().str.casefold()
    return (~normalized.isin(excluded) & normalized.ne("")).to_numpy()


def _unknown_checkpoint_labels(model, labels, class_name: str) -> list[str]:
    """Return stable labels that a checkpoint would otherwise coerce to unknown."""
    decoders = getattr(model, "label_decoders", {})
    decoder = decoders.get(class_name) if isinstance(decoders, dict) else None
    if not isinstance(decoder, dict):
        raise KeyError(f"Checkpoint has no decoder for {class_name}")
    known = {str(value) for value in decoder.values()}
    observed = {str(value) for value in labels.astype(str).unique()}
    return sorted(observed - known)


def _common_metadata(
    args: argparse.Namespace,
    embedding_key: str,
    adata=None,
    training_filter: dict | None = None,
) -> Path:
    metadata = {
        "mode": args.mode,
        "notebook_provenance": args.notebook_provenance,
        "input": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "embedding_key": embedding_key,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_name": Path(args.checkpoint).name,
        "seed": args.seed,
        "batch_key": args.batch_key,
        "cell_type_key": args.cell_type_key,
        "settings": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "embed_how": args.embed_how,
            "embed_max_len": args.embed_max_len,
            "use_knn": args.use_knn,
            "rebuild_knn_graph_requested": getattr(args, "rebuild_knn_graph", False),
            "legacy_detached_mmd": args.legacy_detached_mmd,
            "restore_best_model": args.restore_best_model,
            "knn_graph_rebuilt": getattr(args, "knn_graph_rebuilt", False),
            "align_checkpoint_genes": args.align_checkpoint_genes,
        },
    }
    if args.mode in {"scprint-mmd", "scprint2-ft"}:
        finetune_settings = _finetune_settings(args)
        metadata["settings"]["finetune"] = {
            "max_len": args.finetune_max_len,
            "num_epochs": args.num_epochs,
            "objective": args.finetune_objective,
            "learn_batches_on": finetune_settings["learn_batches_on"],
            "do_mmd_on": finetune_settings["do_mmd_on"],
            "mmd_target": args.mmd_target,
            "mmd_batch_key": args.batch_key,
            "lr": 0.0002,
            "ft_mode": "xpressor",
            "frac_train": 0.8,
            "loss_scalers": finetune_settings["loss_scalers"],
            "use_knn": args.use_knn,
            "classification_targets": _classification_targets(args),
            "organism_classification": bool(
                getattr(args, "include_organism_classification", False)
            ),
            "organism_decoder_trainable": False,
            "knn_forward_enabled": args.use_knn,
            "knn_cell_selector": (
                "six_nearest_stored_scanpy_distance_edges_within_split"
                if args.use_knn
                else None
            ),
            "validation_batch_order": (
                "fixed_randomized_for_mmd"
                if finetune_settings["do_mmd_on"] is not None
                else "source_order"
            ),
        }
    if adata is not None:
        metadata["cells"] = int(adata.n_obs)
        metadata["genes"] = int(adata.n_vars)
        if "checkpoint_gene_alignment" in adata.uns:
            metadata["checkpoint_gene_alignment"] = adata.uns[
                "checkpoint_gene_alignment"
            ]
        if "pandora_cell_filter" in adata.uns:
            metadata["physical_filter"] = adata.uns["pandora_cell_filter"]
        if "scprint_knn_cells" in adata.uns:
            metadata["knn_cells_preprocessing"] = adata.uns["scprint_knn_cells"]
    if training_filter is not None:
        metadata["training_filter"] = training_filter
    metadata_path = Path(args.output).with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_default) + "\n"
    )
    return metadata_path


def _load_scprint(checkpoint: str):
    import torch

    from scprint2 import scPRINT2

    model = scPRINT2.load_from_checkpoint(
        checkpoint, precpt_gene_emb=None, gene_pos_file=None
    )
    model.mask_zeros = False
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        model = model.to(torch.float32)
    return model


def _mmd_class_token(args: argparse.Namespace) -> str:
    # ``mmd_target`` is compatibility-only provenance. Executable groups come
    # from ``batch_key``; the regularized representation stays the biological
    # cell-type token rather than the organism classifier itself.
    return args.cell_type_key


def _classification_targets(args: argparse.Namespace) -> list[str]:
    targets = [args.cell_type_key]
    if getattr(args, "include_organism_classification", False):
        organism_key = "organism_ontology_term_id"
        if organism_key not in targets:
            targets.append(organism_key)
    return targets


def _finetune_settings(args: argparse.Namespace) -> dict:
    if args.finetune_objective == "cell-type-only":
        return {
            "learn_batches_on": None,
            "do_mmd_on": None,
            "loss_scalers": {
                "expr": 0,
                "class": 1,
                "mmd": 0,
                "kl": 0,
                "organism_ontology_term_id": 0,
            },
        }
    mmd_scale = float(args.mmd_scale)
    if mmd_scale < 0:
        raise ValueError("--mmd-scale must be non-negative")
    include_organism = getattr(args, "include_organism_classification", False)
    return {
        "learn_batches_on": "organism_ontology_term_id",
        "do_mmd_on": _mmd_class_token(args),
        "loss_scalers": {
            "expr": 1,
            "class": 1,
            "mmd": mmd_scale,
            "kl": 0.5,
            "organism_ontology_term_id": 1 if include_organism else 0,
        },
    }


def _align_to_checkpoint_genes(adata, model):
    """Zero-pad active-organism genes and enforce checkpoint-local ordering."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    if not isinstance(model._genes, dict):
        raise TypeError("Checkpoint gene alignment requires per-organism vocabularies")
    organisms = list(pd.unique(adata.obs["organism_ontology_term_id"].astype(str)))
    if len(organisms) != 1:
        raise ValueError("Checkpoint alignment currently requires one feature organism")
    organism = organisms[0]
    if organism not in model._genes:
        raise ValueError(f"Checkpoint does not support {organism}")
    expected = list(model._genes[organism])
    expected_positions = {gene: index for index, gene in enumerate(expected)}
    observed = [str(gene) for gene in adata.var_names]
    overlap = [gene for gene in observed if gene in expected_positions]
    if not overlap:
        raise RuntimeError("No input genes overlap the checkpoint vocabulary")
    selected = adata[:, overlap].X
    if not sp.issparse(selected):
        selected = sp.csr_matrix(selected)
    selected = selected.tocoo()
    columns = [expected_positions[overlap[index]] for index in selected.col]
    matrix = sp.csr_matrix(
        (selected.data, (selected.row, columns)),
        shape=(adata.n_obs, len(expected)),
    )
    aligned = ad.AnnData(
        X=matrix,
        obs=adata.obs.copy(),
        var=pd.DataFrame(
            {
                "ensembl_gene_id": expected,
                "organism": organism,
            },
            index=pd.Index(expected, name="ensembl_gene_id"),
        ),
        uns=adata.uns.copy(),
        obsm={key: value.copy() for key, value in adata.obsm.items()},
        obsp={key: value.copy() for key, value in adata.obsp.items()},
    )
    aligned.uns["checkpoint_gene_alignment"] = {
        "organism": organism,
        "input_genes": adata.n_vars,
        "overlap_genes": len(overlap),
        "checkpoint_genes": len(expected),
        "missing_genes_zero_filled": len(expected) - len(overlap),
    }
    return aligned


def _ensure_historical_knn_graph(
    adata, seed: int = 42, force_rebuild: bool = False
) -> bool:
    """Build metacell neighbors with the same expression PCA used by scIB."""
    has_graph = "connectivities" in adata.obsp and "distances" in adata.obsp
    if has_graph and not force_rebuild:
        return False

    import scanpy as sc

    if force_rebuild:
        for key in ("connectivities", "distances"):
            if key in adata.obsp:
                del adata.obsp[key]
        for key in ("neighbors", "pca", "scprint_knn_cells"):
            if key in adata.uns:
                del adata.uns[key]
        if "X_pca" in adata.obsm:
            del adata.obsm["X_pca"]
        if "PCs" in adata.varm:
            del adata.varm["PCs"]

    raw_expression = adata.X.copy()
    had_log1p_metadata = "log1p" in adata.uns
    previous_log1p_metadata = copy.deepcopy(adata.uns.get("log1p"))
    try:
        sc.pp.normalize_total(adata, target_sum=10_000)
        sc.pp.log1p(adata)
        n_components = min(50, adata.n_obs - 1, adata.n_vars - 1)
        if n_components < 2:
            raise RuntimeError("KNN-cell PCA requires at least three cells and genes")
        sc.tl.pca(
            adata,
            n_comps=n_components,
            svd_solver="arpack",
            use_highly_variable=False,
            random_state=seed,
        )
        sc.pp.neighbors(adata, use_rep="X_pca", random_state=seed)
    finally:
        adata.X = raw_expression
        if had_log1p_metadata:
            adata.uns["log1p"] = previous_log1p_metadata
        else:
            adata.uns.pop("log1p", None)
    adata.uns["scprint_knn_cells"] = {
        "source": "expression.X_before_checkpoint_alignment",
        "normalization": "normalize_total_target_sum_10000_then_log1p",
        "pca_dimension": n_components,
        "pca_svd_solver": "arpack",
        "use_highly_variable": False,
        "neighbors_use_rep": "X_pca",
        "knn_cell_selector": "six_nearest_stored_scanpy_distance_edges_within_split",
        "seed": seed,
        "forced_rebuild": force_rebuild,
    }
    return True


def run_scprint(args: argparse.Namespace) -> Task3RunArtifacts:
    config = Task3Config(**vars(args))
    seed_task3(config.seed)
    validate_task3_output_paths(config)
    adata = load_task3_dataset(config)
    model = load_task3_checkpoint(config)
    adata, knn_graph_rebuilt = prepare_task3_dataset(adata, model, config)
    model, training_filter, _ = finetune_task3_checkpoint(model, adata, config)
    embedded = embed_task3(model, adata, config)
    return save_task3_embedding(
        embedded,
        adata,
        config,
        training_filter=training_filter,
        knn_graph_rebuilt=knn_graph_rebuilt,
    )


def run_transcriptformer(args: argparse.Namespace) -> Task3RunArtifacts:
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    output = Path(args.output)
    _refuse_overwrite(output, output.with_suffix(".metadata.json"))
    prepared = output.with_name(output.stem + "_input.h5ad")
    adata = sc.read_h5ad(args.input)
    if "ensembl_id" not in adata.var:
        if "ensembl_gene_id" not in adata.var:
            raise KeyError("Expected var['ensembl_gene_id'] for TranscriptFormer")
        adata.var["ensembl_id"] = adata.var["ensembl_gene_id"].astype(str).to_numpy()
    if args.feature_organism is not None:
        adata.obs["organism_ontology_term_id"] = args.feature_organism
    adata.obs["assay"] = "unknown"
    obs_name_key = "_transcriptformer_input_obs_name"
    adata.obs[obs_name_key] = adata.obs_names.astype(str)
    sample = adata.X[: min(128, adata.n_obs)]
    values = sample.data if sp.issparse(sample) else np.asarray(sample).ravel()
    if values.size and (
        np.nanmin(values) < 0
        or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6)
    ):
        raise ValueError("TranscriptFormer requires raw non-negative integer counts")
    if prepared.exists():
        prepared_adata = sc.read_h5ad(prepared, backed="r")
        if (
            prepared_adata.n_obs != adata.n_obs
            or not prepared_adata.obs_names.equals(adata.obs_names)
            or obs_name_key not in prepared_adata.obs
        ):
            raise RuntimeError(f"Stale TranscriptFormer input artifact: {prepared}")
    else:
        adata.raw = None
        adata.write_h5ad(prepared, compression="lzf")

    command = [
        str(Path(sys.executable).with_name("transcriptformer")),
        "inference",
        "--checkpoint-path",
        args.checkpoint,
        "--data-file",
        str(prepared),
        "--output-path",
        str(output.parent),
        "--output-filename",
        output.name,
        "--gene-col-name",
        "ensembl_id",
        "--use-raw",
        "False",
        "--emb-type",
        "cell",
        "--device",
        "cuda",
        "--num-gpus",
        "1",
        "--precision",
        "16-mixed",
        "--batch-size",
        str(args.batch_size),
        "--oom-dataloader",
        "--n-data-workers",
        str(args.num_workers),
    ]
    subprocess.run(command, check=True)
    result = sc.read_h5ad(output)
    if result.n_obs != adata.n_obs:
        raise RuntimeError("TranscriptFormer changed the number of cells")
    if not result.obs_names.equals(adata.obs_names):
        if obs_name_key not in result.obs or not np.array_equal(
            result.obs[obs_name_key].astype(str).to_numpy(),
            adata.obs_names.astype(str).to_numpy(),
        ):
            raise RuntimeError("TranscriptFormer changed cell order")
        result.obs_names = adata.obs_names.copy()
    if "embeddings" not in result.obsm:
        raise RuntimeError("TranscriptFormer output has no obsm['embeddings']")
    result.obsm["model_emb"] = np.asarray(result.obsm["embeddings"], dtype=np.float32)
    result.write_h5ad(output, compression="lzf")
    metadata_path = _common_metadata(args, "model_emb")
    return Task3RunArtifacts(output=output, metadata=metadata_path)


def run_task3(config: Task3Config) -> Task3RunArtifacts:
    """Execute a Task3 configuration without exposing command-line machinery."""
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DISABLED", "true")
    args = config.as_namespace()
    if config.mode == "transcriptformer":
        return run_transcriptformer(args)
    return run_scprint(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=TASK3_MODES,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-key", default="orig.ident")
    parser.add_argument("--cell-type-key", default="cell_type_ontology_term_id")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--embed-how", default="random expr")
    parser.add_argument("--embed-max-len", type=int, default=2200)
    parser.add_argument("--finetune-max-len", type=int, default=2200)
    parser.add_argument(
        "--use-knn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use neighboring cells for metacell-style scPRINT inputs. The historical "
            "notebooks inherited Embedder(use_knn=True); task3 runners default to the "
            "previous cell-by-cell behavior unless this flag is supplied."
        ),
    )
    parser.add_argument(
        "--rebuild-knn-graph",
        action="store_true",
        help=(
            "Discard any stored PCA/neighbors graph and rebuild it from raw X "
            "with normalize_total(1e4), log1p, PCA, and neighbors(use_rep='X_pca')."
        ),
    )
    parser.add_argument(
        "--legacy-detached-mmd",
        action="store_true",
        help=(
            "Reproduce the historical fine-tuning bug where the legacy MMD "
            "statistic was converted to a float before being added to the loss."
        ),
    )
    parser.add_argument(
        "--restore-best-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the lowest-validation-loss epoch after fine-tuning.",
    )
    parser.add_argument(
        "--finetune-objective",
        choices=("reconstruction-mmd", "cell-type-only"),
        default="reconstruction-mmd",
        help=(
            "Fine-tuning losses. cell-type-only disables expression, KL, MMD, "
            "organism supervision, and learned batch embeddings."
        ),
    )
    parser.add_argument(
        "--mmd-target",
        choices=("label", "species"),
        default="species",
        help=(
            "Biological intent recorded in provenance. The actual MMD groups are "
            "the values of --batch-key; the regularized representation is the "
            "cell-type token."
        ),
    )
    parser.add_argument(
        "--mmd-scale",
        type=float,
        default=0.03,
        help=(
            "Coefficient applied to the differentiable MMD term for the "
            "reconstruction-mmd objective."
        ),
    )
    parser.add_argument(
        "--include-organism-classification",
        action="store_true",
        help=(
            "Also supervise organism_ontology_term_id during fine-tuning. This "
            "reproduces the behavior of the fresh two-species task3 run that "
            "produced the 0.667799 historical-scIB total."
        ),
    )
    parser.add_argument(
        "--exclude-train-label",
        action="append",
        default=["unknown"],
        help="Case-insensitive cell-type label to exclude from fine-tuning.",
    )
    parser.add_argument(
        "--include-unknown-train-labels",
        action="store_true",
        help=(
            "Do not apply the runner's default unknown-label exclusion. The "
            "historical task3 notebook fine-tuned on all 27,200 cells."
        ),
    )
    parser.add_argument(
        "--require-known-train-labels",
        action="store_true",
        help=(
            "Fail before fine-tuning if any requested class is absent from the "
            "checkpoint decoder instead of silently converting it to unknown."
        ),
    )
    parser.add_argument(
        "--feature-organism",
        help="Override the feature organism for TranscriptFormer input preparation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notebook-provenance", required=True)
    parser.add_argument("--align-checkpoint-genes", action="store_true")
    args = parser.parse_args()
    run_task3(Task3Config(**vars(args)))


if __name__ == "__main__":
    main()
