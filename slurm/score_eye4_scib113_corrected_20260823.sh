#!/bin/bash
#SBATCH --job-name=eye4-scib113-corrected
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/scib113-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/scib113-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_corrected_20260823}"
SOURCE="${ROOT}/eye4_mouse_ortholog_scprint_preprocessed.h5ad"
BASELINES="${ROOT}/embeddings/expression_pca50_random_seed42.h5ad"
ZERO_SHOT="${ROOT}/embeddings/scprint2_zero_shot_small_v2_native_knn.h5ad"
FINE_TUNED="${ROOT}/embeddings/scprint2_ft_small_v2_native_mmd003_knn.h5ad"
TRANSCRIPTFORMER="${ROOT}/embeddings/transcriptformer_metazoa_mouse_ortholog.h5ad"
OUTPUT="${ROOT}/scib113/eye4_scib113_corrected_fullmetrics_scores.csv"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

for path in "$SOURCE" "$BASELINES" "$ZERO_SHOT" "$FINE_TUNED" "$TRANSCRIPTFORMER"; do
  test -s "$path"
done
test -f "${ROOT}/embeddings/SCPRINT2_NATIVE.COMPLETE"
test -f "${ROOT}/embeddings/BASELINES_TRANSCRIPTFORMER.COMPLETE"
test ! -e "$OUTPUT"
test ! -e "${OUTPUT%.csv}.COMPLETE"

srun "$PYTHON" "${REPO_ROOT}/scripts/benchmark_task3_scib113_precomputed.py" \
  --source "$SOURCE" \
  --batch-key species \
  --label-key celltype \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "PCA=${BASELINES}:X_pca" \
  --embedding "Random-seed42=${BASELINES}:random" \
  --embedding "scPRINT-2-ZS-native-cell-token=${ZERO_SHOT}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-FT-native-cell-token-MMD003=${FINE_TUNED}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "TranscriptFormer-Metazoa-mouse-ortholog=${TRANSCRIPTFORMER}:model_emb" \
  --output "$OUTPUT"

test -s "$OUTPUT"
test -s "${OUTPUT%.csv}.metadata.json"
test -s "${OUTPUT%.csv}.COMPLETE"
