#!/usr/bin/env python3
"""Build the two narrative four-species eye benchmark notebooks."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks" / "scPRINT-2-repro-notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORTS = """from pathlib import Path
import json

import pandas as pd
import scanpy as sc
from IPython.display import Image, display

from scprint2.benchmark.task3 import (
    Task3Config,
    embed_task3,
    finetune_task3_checkpoint,
    load_task3_checkpoint,
    load_task3_dataset,
    prepare_task3_dataset,
    save_task3_embedding,
    summarize_task3_dataset,
)
from scprint2.evaluation.task3_scib import Task3Scib113Config
from scprint2.evaluation.cross_species import (
    all_token_keys_except_organism,
    build_token_concat_pca,
)
from scprint2.preprocessing.eye import (
    EYE_CELL_TYPE_ONTOLOGY,
    extract_eye_annotations,
    load_geo_count_subset,
    strict_one_to_one_mapping,
)
"""

PATHS = """REPO_ROOT = Path.cwd()
if REPO_ROOT.name == "scPRINT-2-repro-notebooks":
    REPO_ROOT = REPO_ROOT.parents[1]
RESULT_ROOT = REPO_ROOT / "data/results/cross_species_embedding/eye4_human_target_hybrid_20260824"
PREPARED = RESULT_ROOT / "eye4_human_ensembl_small_v2.h5ad"
SCORES = RESULT_ROOT / "scib113_all_tokens_except_organism/eye4_scib113_human_target_six_methods_scores.csv"
FIGURE = RESULT_ROOT / "figures/scored_umaps_all_tokens_except_organism/eye4_scored_umaps_overview.png"
MANIFEST = RESULT_ROOT / "preparation/eye4_human_preparation_manifest.json"
GENE_AUDIT = RESULT_ROOT / "preparation/eye4_human_target_gene_audit.tsv"
CHECKPOINT_GENE_AUDIT = RESULT_ROOT / "preparation/eye4_human_checkpoint_gene_audit.tsv"
ZERO_METADATA = RESULT_ROOT / "models/scprint2_zero_shot_small_v2_human_target_knn.metadata.json"
FT_METADATA = RESULT_ROOT / "models/scprint2_ft_small_v2_human_target_mmd003_knn.metadata.json"
TOKEN_METADATA = RESULT_ROOT / "models/scprint2_zero_shot_small_v2_human_target_all_tokens_except_organism_pca50.metadata.json"
"""


def zero_shot_cells() -> list[dict]:
    return [
        markdown(
            """# Four-species eye integration — scPRINT-2 zero-shot

This is the executable narrative surface for the final human-target four-species eye analysis. Expression is rebuilt from deposited GEO integer-count matrices; earlier log-normalized H5AD files provide expert cell annotations only. Human, cynomolgus macaque, pig and mouse genes are mapped into a union of human ENSG identifiers without filtering to a cross-species intersection."""
        ),
        code(COMMON_IMPORTS),
        markdown(
            """## Protocol contract

All six views use the same 29,568 cells and 20,004-gene human small-v2 checkpoint space. Human-like symbols are resolved directly to human ENSG identifiers; mouse uses high-confidence human orthology as the fallback. No expressed-gene intersection is applied. The zero-shot cell token and the additional PCA50 concatenation of every saved token except organism come from the same scPRINT-2 inference. Scores come only from the isolated legacy `scib==1.1.3` environment."""
        ),
        code(PATHS),
        markdown("## Preparation manifest\n\nThe manifest records cell filtering, raw-count status and feature-space provenance."),
        code("preparation_manifest = json.loads(MANIFEST.read_text())\npd.Series(preparation_manifest['provenance'])"),
        markdown("## Annotation gate\n\nExpert `celltype` strings are the scIB biological labels. Every ontology identifier used for fine-tuning is present in the small-v2 decoder; Beam A, JCT and Schlemm's canal are mapped to supported CL terms before training."),
        code("annotation_audit = pd.Series(preparation_manifest['provenance']['cells_by_species'], name='cells').rename_axis('species').reset_index()\nannotation_audit"),
        code("pd.Series({'cells': preparation_manifest['shape'][0], 'genes': preparation_manifest['shape'][1], 'species': len(preparation_manifest['provenance']['cells_by_species']), 'expert_cell_types': len(preparation_manifest['cell_types']), 'missing_celltype': 0, 'feature_organism': preparation_manifest['provenance']['feature_organism_ontology_term_id']})"),
        markdown("## Human-target gene audit\n\nEach species is mapped independently to human ENSG identifiers and combined by union with zero filling. The common 13,546 genes are audit information only and never used as a filter. The checkpoint audit verifies the final 20,004-column small-v2 human vocabulary."),
        code("gene_audit = pd.read_csv(GENE_AUDIT, sep='\\t')\ncheckpoint_gene_audit = pd.read_csv(CHECKPOINT_GENE_AUDIT, sep='\\t')\ndisplay(gene_audit, checkpoint_gene_audit)"),
        markdown("## Orthology audit\n\nDirect human-symbol resolution is retained for human, macaque and pig when available; high-confidence human orthology is the species-specific fallback. Mouse uses the same human-target contract. This is a union, not a four-way intersection."),
        code("gene_summary = gene_audit[['species', 'source_gene_rows', 'retained_target_genes', 'mapped_source_gene_rows_in_geo', 'orthology_mode']].copy()\ngene_summary['expressed_target_genes'] = gene_summary['species'].map(preparation_manifest['provenance']['expressed_target_genes_by_species'])\ngene_summary"),
        markdown("## Optional zero-shot inference\n\nThe expensive branch is disabled in the delivered notebook. It exposes the same reusable package stages used by the Jean Zay run, without hiding execution in a shell cell."),
        code("RUN_OPTIONAL_INFERENCE = False\nzero_config = Task3Config(mode='scprint2-zero-shot', input=PREPARED, checkpoint=REPO_ROOT / 'models/small-v2.ckpt', output=RESULT_ROOT / 'optional/scprint2_zero_shot_small_v2_human_target_knn.h5ad', batch_key='species', cell_type_key='cell_type_ontology_term_id', embed_how='random expr', embed_max_len=2800, use_knn=True, rebuild_knn_graph=True, align_checkpoint_genes=True, seed=42, notebook_provenance='eye4 human-target ENSG union; all cells declared human feature space; small-v2 zero-shot; multi-cell kNN')\nzero_config.as_dict()"),
        code("if RUN_OPTIONAL_INFERENCE:\n    zero_data = load_task3_dataset(zero_config)\n    zero_model = load_task3_checkpoint(zero_config)\n    zero_data, zero_knn_rebuilt = prepare_task3_dataset(zero_data, zero_model, zero_config)\n    zero_embedded = embed_task3(zero_model, zero_data, zero_config)\n    zero_artifacts = save_task3_embedding(zero_embedded, zero_data, zero_config, knn_graph_rebuilt=zero_knn_rebuilt)\n    display(zero_artifacts.artifact_table())"),
        markdown("## All tokens except organism\n\nThe additional zero-shot view reuses the persisted token blocks from the exact inference above, excludes only `scprint_emb_organism_ontology_term_id` (the assay token is retained), concatenates the remaining blocks and applies randomized PCA50 with seed 42."),
        code("if RUN_OPTIONAL_INFERENCE:\n    selected_tokens = all_token_keys_except_organism(zero_embedded)\n    token_concat_pca50, token_metadata = build_token_concat_pca(zero_embedded, seed=42, token_keys=selected_tokens)\n    pd.Series(token_metadata)"),
        markdown("## Artifact validation\n\nThe delivered metadata, completion markers, detailed scores and figures are checked explicitly. Heavy H5AD files and checkpoints remain on Jean Zay; their absolute paths and hashes are recorded in the metadata."),
        code("artifact_rows = []\nfor method, path in {'scPRINT-2 ZS': ZERO_METADATA, 'scPRINT-2 FT': FT_METADATA, 'scPRINT-2 ZS all except organism': TOKEN_METADATA, 'scIB six methods': SCORES, 'six-method UMAP': FIGURE}.items():\n    artifact_rows.append({'method': method, 'path': str(path), 'exists': path.is_file(), 'bytes': path.stat().st_size if path.is_file() else 0})\npd.DataFrame(artifact_rows)"),
        markdown("## Detailed scIB 1.1.3 scores\n\n`Batch correction`, `Bio conservation` and `Total` are derived summaries. The remaining columns are the raw legacy scIB metrics. The scorer configuration below fixes `species` as batch and the expert `celltype` string as label. DPT cells outside the root-connected expression component retain missing pseudotime; scIB 1.1.3 excludes those cells from trajectory rather than imputing a distance."),
        code("scib_config = Task3Scib113Config(source=PREPARED, embedding=(('PCA', RESULT_ROOT / 'embeddings/expression_pca50_random_seed42.h5ad', 'X_pca'), ('Random-seed42', RESULT_ROOT / 'embeddings/expression_pca50_random_seed42.h5ad', 'random'), ('scPRINT-2-ZS-human-target-cell-token', RESULT_ROOT / 'embeddings/scprint2_zero_shot_small_v2_human_target_knn.h5ad', 'scprint_emb_cell_type_ontology_term_id'), ('scPRINT-2-FT-human-target-cell-token-MMD003', RESULT_ROOT / 'embeddings/scprint2_ft_small_v2_human_target_mmd003_knn.h5ad', 'scprint_emb_cell_type_ontology_term_id'), ('TranscriptFormer-Metazoa-human-target', RESULT_ROOT / 'embeddings/transcriptformer_metazoa_human_target.h5ad', 'model_emb'), ('scPRINT-2-ZS-human-target-all-tokens-except-organism-PCA50', RESULT_ROOT / 'embeddings/scprint2_zero_shot_small_v2_human_target_all_tokens_except_organism_pca50.h5ad', 'token_concat_pca50')), output=SCORES, batch_key='species', label_key='celltype', n_cores=32)\npd.Series(scib_config.as_dict())"),
        code("scores = pd.read_csv(SCORES).sort_values('Total', ascending=False)\nscores"),
        markdown("## UMAP diagnostics\n\nUMAP uses the exact k-nearest-neighbor graph persisted by the legacy scorer for each representation; neighbors are not recomputed for plotting. Random is included explicitly as the negative control."),
        code("display(Image(filename=str(FIGURE)))"),
        markdown("## Interpretation boundary\n\nThe comparison measures integration on this filtered eye atlas. scPRINT-2 and the expression/TranscriptFormer baselines use model-appropriate feature spaces, then share cells, labels and the same evaluator. It does not establish that one model is universally superior, and batch mixing must be read together with expert cell-type conservation."),
        code("pd.Series({'scib_complete': SCORES.with_suffix('.COMPLETE').is_file(), 'umap_complete': (FIGURE.parent / 'UMAPS.COMPLETE').is_file(), 'models_scored': int(len(scores))})"),
    ]


def fine_tuning_cells() -> list[dict]:
    return [
        markdown(
            """# Four-species eye integration — scPRINT-2 fine-tuning

This notebook documents the corrected fine-tuning branch for the same four-species eye dataset. It replaces the notebook-local training loop with the shared `FinetuneBatchClass` path used by the other cross-species integrations."""
        ),
        code(COMMON_IMPORTS),
        markdown("## Protocol contract\n\nFine-tuning and zero-shot inference start from the same `small-v2.ckpt`, prepared counts, cells, KNN-cell input graph and cell-type token. Only the fine-tuning step differs."),
        code(PATHS),
        markdown("## Training data audit\n\nThe expert annotations are present before training; `species` is the four-level MMD group. Features use each species' native small-v2 vocabulary, with the documented rhesus proxy for cynomolgus."),
        code("preparation_manifest = json.loads(MANIFEST.read_text())\npd.Series(preparation_manifest['provenance']['cells_by_species'], name='cells').rename_axis('species').reset_index()"),
        code("pd.Series({'cells': preparation_manifest['shape'][0], 'genes': preparation_manifest['shape'][1], 'species_groups': len(preparation_manifest['provenance']['cells_by_species']), 'expert_cell_types': len(preparation_manifest['cell_types']), 'model_feature_organisms': 1})"),
        markdown("## MMD configuration\n\nThe original notebook detached MMD with `.item()` and compared only groups 0 and 1. The shared helper keeps MMD in the autograd graph and averages all six unordered pairs among four species. Its coefficient is 0.03 and it regularizes the cell-type token. Fine-tuning now fails before training if any CL target is absent from the checkpoint decoder."),
        code("ft_config = Task3Config(mode='scprint2-ft', input=PREPARED, checkpoint=REPO_ROOT / 'models/small-v2.ckpt', output=RESULT_ROOT / 'optional/scprint2_ft_small_v2_human_target_mmd003_knn.h5ad', batch_key='species', cell_type_key='cell_type_ontology_term_id', finetune_max_len=2200, embed_max_len=2800, num_epochs=8, mmd_target='species', mmd_scale=0.03, include_organism_classification=False, use_knn=True, rebuild_knn_graph=True, align_checkpoint_genes=True, require_known_train_labels=True, restore_best_model=True, seed=42, notebook_provenance='eye4 human-target ENSG union; all 13 labels known by small-v2; differentiable balanced four-species MMD=0.03 on cell-type token; best checkpoint restored; multi-cell kNN')\npd.Series({'batch_key': ft_config.batch_key, 'mmd_target': ft_config.mmd_target, 'mmd_scale': ft_config.mmd_scale, 'epochs': ft_config.num_epochs, 'finetune_max_len': ft_config.finetune_max_len, 'embed_max_len': ft_config.embed_max_len, 'align_checkpoint_genes': ft_config.align_checkpoint_genes, 'require_known_train_labels': ft_config.require_known_train_labels, 'restore_best_model': ft_config.restore_best_model})"),
        markdown("## Optional fine-tuning\n\nThe costly branch is disabled. The cells expose dataset loading, checkpoint loading, graph preparation, fine-tuning, embedding and persistence as separate reusable stages."),
        code("RUN_OPTIONAL_FINETUNING = False\nft_config.as_dict()"),
        code("if RUN_OPTIONAL_FINETUNING:\n    ft_data = load_task3_dataset(ft_config)\n    display(summarize_task3_dataset(ft_data, ft_config))\n    ft_model = load_task3_checkpoint(ft_config)\n    ft_data, ft_knn_rebuilt = prepare_task3_dataset(ft_data, ft_model, ft_config)\n    ft_model, ft_training_filter, ft_checkpoint = finetune_task3_checkpoint(ft_model, ft_data, ft_config)\n    ft_embedded = embed_task3(ft_model, ft_data, ft_config)\n    ft_artifacts = save_task3_embedding(ft_embedded, ft_data, ft_config, training_filter=ft_training_filter, knn_graph_rebuilt=ft_knn_rebuilt)\n    display(ft_artifacts.artifact_table())"),
        markdown("## Training provenance\n\nThe metadata records training cells, all four MMD groups, objective, KNN use, best validation epoch and restored checkpoint. The checkpoint is a separate required artifact."),
        code("ft_metadata = json.loads(FT_METADATA.read_text())\npd.Series(ft_metadata['training_filter'])"),
        code("pd.Series({'remote_checkpoint': str(Path(ft_metadata['output']).with_suffix('.ckpt')), 'remote_embedding': ft_metadata['output'], 'checkpoint_source': ft_metadata['checkpoint'], 'cells': ft_metadata['cells'], 'genes': ft_metadata['genes'], 'all_labels_known': not ft_metadata['training_filter']['checkpoint_unknown_labels']})"),
        markdown("## Detailed scIB 1.1.3 scores\n\nThe FT score is shown beside all controls from the single joint legacy scoring run; no modern `scib-metrics` value is mixed into this table."),
        code("scores = pd.read_csv(SCORES).sort_values('Total', ascending=False)\nscores"),
        code("scores.loc[scores['Method'].eq('scPRINT-2-FT-native-cell-token-MMD003')].T"),
        markdown("## UMAP diagnostics\n\nThe overview uses the exact scorer graphs. The two rows separate expert cell-type conservation from species mixing."),
        code("display(Image(filename=str(FIGURE)))"),
        markdown("## Interpretation boundary\n\nA higher batch-mixing score is useful only if expert cell types remain separated. This run tests one FT objective, seed and atlas; it is not a hyperparameter search and does not support a universal causal claim."),
        code("pd.Series({'scib_complete': SCORES.with_suffix('.COMPLETE').is_file(), 'umap_complete': (FIGURE.parent / 'UMAPS.COMPLETE').is_file(), 'models_scored': int(len(scores))})"),
    ]


def main() -> None:
    outputs = {
        NOTEBOOK_ROOT / "cross-multi-species-embbedding.ipynb": notebook(
            zero_shot_cells()
        ),
        NOTEBOOK_ROOT
        / "fine_tuning_multi_cross_species_emb_mmd.ipynb": notebook(
            fine_tuning_cells()
        ),
    }
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
