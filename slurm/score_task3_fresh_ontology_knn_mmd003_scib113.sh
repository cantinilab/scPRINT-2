#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=slurm/task3-fresh-scib113-%j.out
#SBATCH --error=slurm/task3-fresh-scib113-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
export R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
RUN_ID="task3_fresh_ontology_knn_mmd003_20260815"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/${RUN_ID}"
ZERO_SHOT="${ROOT}/scprint2_zero_shot_small_v2_knn.h5ad"
FINE_TUNED="${ROOT}/scprint2_ft_small_v2_mmd003_knn.h5ad"
ZERO_SHOT_CONCAT="${ROOT}/scprint2_zero_shot_all_tokens_except_assay_organism_pca50.h5ad"
OUTPUT="${ROOT}/task3_scib113_precomputed_scores.csv"

test -f "$ROOT/EMBEDDINGS_COMPLETE"
test -s "$ZERO_SHOT"
test -s "$FINE_TUNED"
test -s "$ZERO_SHOT_CONCAT"

srun "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_scib113_precomputed.py" \
  --source "$SOURCE" \
  --batch-key orig.ident \
  --label-key cell_type_ontology_term_id \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "scPRINT-2-FT-cell-token-mmd003-knn=${FINE_TUNED}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-ZS-cell-token-knn=${ZERO_SHOT}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-ZS-all-tokens-except-assay-organism-pca50=${ZERO_SHOT_CONCAT}:token_concat_pca50" \
  --output "$OUTPUT"
