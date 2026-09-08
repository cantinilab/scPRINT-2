#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scib113-one-method-%j.out
#SBATCH --error=slurm/pandora-scib113-one-method-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
export R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

METHOD="${PANDORA_SCIB_METHOD:?Set PANDORA_SCIB_METHOD}"
SOURCE="${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad"
OBS_NAME_ARGS=()

case "$METHOD" in
  ft-cell)
    NAME="scPRINT-2-FT-cell-token-mmd003-no-organism-classification-knn"
    FILE="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_ft_mmd003_no_org_cls_knn.h5ad"
    KEY="scprint_emb_cell_type_ontology_term_id"
    ;;
  ft-all)
    NAME="scPRINT-2-FT-all-tokens-mmd003-no-organism-classification-knn"
    FILE="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_ft_mmd003_no_org_cls_knn_token_concat_pca50.h5ad"
    KEY="token_concat_pca50"
    ;;
  zs-cell)
    NAME="scPRINT-2-ZS-cell-token-knn"
    FILE="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_zs_knn_fixed.h5ad"
    KEY="scprint_emb_cell_type_ontology_term_id"
    ;;
  zs-all)
    NAME="scPRINT-2-ZS-all-tokens-pca50"
    FILE="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_zs_knn_fixed_token_concat_pca50.h5ad"
    KEY="token_concat_pca50"
    ;;
  transcriptformer)
    NAME="TranscriptFormer"
    FILE="${DATA_ROOT}/pandora_lung_transcriptformer.h5ad"
    KEY="model_emb"
    OBS_NAME_ARGS=(
      --obs-name-replace
      ".mouse_homolog_ensmusg=.human_one2one"
    )
    ;;
  *)
    echo "Unknown PANDORA_SCIB_METHOD: $METHOD" >&2
    exit 2
    ;;
esac

OUTPUT="${DATA_ROOT}/pandora_lung_filtered_93423_scib113_${METHOD}.csv"

cd "$REPO_ROOT"
test -s "$SOURCE"
test -s "$FILE"

srun "$PYTHON" scripts/benchmark_task3_scib113_precomputed.py \
  --source "$SOURCE" \
  --batch-key species \
  --label-key cell_type_ontology_term_id \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  "${OBS_NAME_ARGS[@]}" \
  --embedding "${NAME}=${FILE}:${KEY}" \
  --output "$OUTPUT"
