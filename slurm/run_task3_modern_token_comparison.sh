#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-modern-token-comparison-%j.out
#SBATCH --error=slurm/task3-modern-token-comparison-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3"
CONCAT="${ROOT}/task3_token_concat_pca50"
OUTPUT="${CONCAT}/task3_modern_token_comparison.csv"

"$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --embedding "scPRINT-1=${ROOT}/scprint1_embeddings.h5ad:scprint_emb" \
  --embedding "scPRINT-1-cell-token=${ROOT}/scprint1_embeddings.h5ad:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-1-token-concat-pca50=${CONCAT}/scprint1_token_concat_pca50.h5ad:token_concat_pca50" \
  --embedding "scPRINT-2=${ROOT}/scprint_zero_shot_embeddings.h5ad:scprint_emb" \
  --embedding "scPRINT-2-cell-token=${ROOT}/scprint_zero_shot_embeddings.h5ad:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-token-concat-pca50=${CONCAT}/scprint2_token_concat_pca50.h5ad:token_concat_pca50" \
  --embedding "scPRINT-2-FT=${ROOT}/scprint_mmd_embeddings.h5ad:scprint_emb" \
  --embedding "scPRINT-2-FT-cell-token=${ROOT}/scprint_mmd_embeddings.h5ad:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-FT-token-concat-pca50=${CONCAT}/scprint2_ft_token_concat_pca50.h5ad:token_concat_pca50" \
  --embedding "TranscriptFormer-Metazoa=${ROOT}/transcriptformer_metazoa_embeddings.h5ad:model_emb"
