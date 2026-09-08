#!/bin/bash
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --array=0-6%3
#SBATCH --output=slurm/pandora-task3-matched-scib113-%A_%a.out
#SBATCH --error=slurm/pandora-task3-matched-scib113-%A_%a.out

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

SOURCE="${PANDORA_INPUT:-${WORK_ROOT}/data/pandora_lung/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad}"
RUN_ID="pandora_task3_matched_20260817"
ROOT="${WORK_ROOT}/data/pandora_lung/${RUN_ID}"
FT="${ROOT}/scprint2_ft_small_v2_mmd003_knn_with_organism_classification.h5ad"
FT_ALL="${ROOT}/scprint2_ft_all_tokens_except_assay_organism_pca50.h5ad"
ZS="${ROOT}/scprint2_zero_shot_small_v2_knn.h5ad"
ZS_ALL="${ROOT}/scprint2_zero_shot_all_tokens_except_assay_organism_pca50.h5ad"
TF="${ROOT}/transcriptformer_metazoa_fresh.h5ad"
BASELINES="${ROOT}/expression_pca_and_random.h5ad"
SCORE_DIR="${ROOT}/scores"
mkdir -p "$SCORE_DIR"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SLUG="ft_cell"
    SPEC="scPRINT-2-FT-cell-token-mmd003-task3-matched=${FT}:scprint_emb_cell_type_ontology_term_id"
    ;;
  1)
    SLUG="ft_all"
    SPEC="scPRINT-2-FT-all-tokens-except-assay-organism-pca50-task3-matched=${FT_ALL}:token_concat_pca50"
    ;;
  2)
    SLUG="zs_cell"
    SPEC="scPRINT-2-ZS-cell-token-task3-matched=${ZS}:scprint_emb_cell_type_ontology_term_id"
    ;;
  3)
    SLUG="zs_all"
    SPEC="scPRINT-2-ZS-all-tokens-except-assay-organism-pca50-task3-matched=${ZS_ALL}:token_concat_pca50"
    ;;
  4)
    SLUG="transcriptformer"
    SPEC="TranscriptFormer-Metazoa-task3-matched=${TF}:model_emb"
    ;;
  5)
    SLUG="pca"
    SPEC="PCA-expression-CP10K-log1p-task3-matched=${BASELINES}:X_pca"
    ;;
  6)
    SLUG="random"
    SPEC="Random-seed42-task3-matched=${BASELINES}:random"
    ;;
  *)
    echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac
OUTPUT="${SCORE_DIR}/${SLUG}.csv"

test -f "$ROOT/SCPRINT2_EMBEDDINGS_COMPLETE"
test -f "$ROOT/TRANSCRIPTFORMER_EMBEDDING_COMPLETE"
test -s "$SOURCE"
if [[ -e "$OUTPUT" || -e "${OUTPUT%.csv}.COMPLETE" ]]; then
  echo "Refusing to overwrite score artifact: $OUTPUT" >&2
  exit 3
fi

cd "$REPO_ROOT"
srun "$PYTHON" scripts/benchmark_task3_scib113_precomputed.py \
  --source "$SOURCE" \
  --batch-key species \
  --label-key cell_type_ontology_term_id \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "$SPEC" \
  --output "$OUTPUT"
