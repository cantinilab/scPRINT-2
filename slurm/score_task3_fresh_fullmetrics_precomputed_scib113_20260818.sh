#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=slurm/task3-fresh-fullmetrics-precomputed-scib113-%j.out
#SBATCH --error=slurm/task3-fresh-fullmetrics-precomputed-scib113-%j.out

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
SCPRINT_ROOT="${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_ontology_knn_mmd003_20260815"
TF_ROOT="${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_transcriptformer_metazoa_scib113_20260817"
RESULT_ROOT="${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_fullmetrics_precomputed_scib113_20260818"
OUTPUT="${RESULT_ROOT}/task3_scib113_precomputed_fullmetrics_scores.csv"

ZERO_SHOT="${SCPRINT_ROOT}/scprint2_zero_shot_small_v2_knn.h5ad"
FINE_TUNED="${SCPRINT_ROOT}/scprint2_ft_small_v2_mmd003_knn.h5ad"
ZERO_SHOT_CONCAT="${SCPRINT_ROOT}/scprint2_zero_shot_all_tokens_except_assay_organism_pca50.h5ad"
TRANSCRIPTFORMER="${TF_ROOT}/transcriptformer_metazoa_fresh.h5ad"

test -s "$SOURCE"
test -s "$ZERO_SHOT"
test -s "$FINE_TUNED"
test -s "$ZERO_SHOT_CONCAT"
test -s "$TRANSCRIPTFORMER"
test ! -e "$OUTPUT"
test ! -e "${OUTPUT%.csv}.COMPLETE"
mkdir -p "$RESULT_ROOT"

srun "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_scib113_precomputed.py" \
  --source "$SOURCE" \
  --batch-key orig.ident \
  --label-key cell_type_ontology_term_id \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "scPRINT-2-FT-cell-token-mmd003-knn-fullmetrics=${FINE_TUNED}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-ZS-cell-token-knn-fullmetrics=${ZERO_SHOT}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-ZS-all-tokens-except-assay-organism-pca50-fullmetrics=${ZERO_SHOT_CONCAT}:token_concat_pca50" \
  --embedding "TranscriptFormer-Metazoa-fullmetrics=${TRANSCRIPTFORMER}:model_emb" \
  --output "$OUTPUT"
