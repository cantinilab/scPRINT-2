#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-modern-scib-%j.out
#SBATCH --error=slurm/task3-modern-scib-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3"
OUTPUT="${ROOT}/task3_modern_notebook/task3_modern_scib.csv"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --embedding "scPRINT-1=${ROOT}/scprint1_embeddings.h5ad:scprint_emb" \
  --embedding "scPRINT-1-cell-token=${ROOT}/scprint1_embeddings.h5ad:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2=${ROOT}/scprint_zero_shot_embeddings.h5ad:scprint_emb" \
  --embedding "scPRINT-2-cell-token=${ROOT}/scprint_zero_shot_embeddings.h5ad:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-FT=${ROOT}/scprint_mmd_embeddings.h5ad:scprint_emb" \
  --embedding "scPRINT-2-FT-cell-token=${ROOT}/scprint_mmd_embeddings.h5ad:scprint_emb_cell_type_ontology_term_id" \
  --embedding "TranscriptFormer-Metazoa=${ROOT}/transcriptformer_metazoa_embeddings.h5ad:model_emb"
