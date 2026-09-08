#!/bin/bash
#SBATCH --job-name=eye4-human-scib113
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/scib113-%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/scib113-%A_%a.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_human_target_hybrid_20260824}"
SOURCE="${ROOT}/eye4_human_ensembl_small_v2.h5ad"
BASELINES="${ROOT}/embeddings/expression_pca50_random_seed42.h5ad"
ZERO_SHOT="${ROOT}/embeddings/scprint2_zero_shot_small_v2_human_target_knn.h5ad"
FINE_TUNED="${ROOT}/embeddings/scprint2_ft_small_v2_human_target_mmd003_knn.h5ad"
TRANSCRIPTFORMER="${ROOT}/embeddings/transcriptformer_metazoa_human_target.h5ad"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

METHODS=(
  "PCA"
  "Random-seed42"
  "scPRINT-2-ZS-human-target-cell-token"
  "scPRINT-2-FT-human-target-cell-token-MMD003"
  "TranscriptFormer-Metazoa-human-target"
)
FILES=(
  "$BASELINES"
  "$BASELINES"
  "$ZERO_SHOT"
  "$FINE_TUNED"
  "$TRANSCRIPTFORMER"
)
KEYS=(
  "X_pca"
  "random"
  "scprint_emb_cell_type_ontology_term_id"
  "scprint_emb_cell_type_ontology_term_id"
  "model_emb"
)

INDEX="${SLURM_ARRAY_TASK_ID}"
METHOD="${METHODS[$INDEX]}"
EMBEDDING="${FILES[$INDEX]}"
KEY="${KEYS[$INDEX]}"
METHOD_ROOT="${ROOT}/scib113/by_method/${METHOD}"
OUTPUT="${METHOD_ROOT}/${METHOD}.csv"

test -s "$SOURCE"
test -s "$EMBEDDING"
test -f "${ROOT}/embeddings/SCPRINT2_HUMAN_TARGET.COMPLETE"
test -f "${ROOT}/embeddings/BASELINES_TRANSCRIPTFORMER_HUMAN.COMPLETE"
mkdir -p "$METHOD_ROOT"
test ! -e "$OUTPUT"
test ! -e "${OUTPUT%.csv}.COMPLETE"

srun "$PYTHON" "${REPO_ROOT}/scripts/benchmark_task3_scib113_precomputed.py" \
  --source "$SOURCE" \
  --batch-key species \
  --label-key celltype \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "${METHOD}=${EMBEDDING}:${KEY}" \
  --output "$OUTPUT"

test -s "$OUTPUT"
test -s "${OUTPUT%.csv}.metadata.json"
test -s "${OUTPUT%.csv}.COMPLETE"
