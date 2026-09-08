#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scib113-mmd003-no-org-cls-knn-%j.out
#SBATCH --error=slurm/pandora-scib113-mmd003-no-org-cls-knn-%j.out

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

SOURCE="${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad"
FT="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_ft_mmd003_no_org_cls_knn.h5ad"
FT_ALL="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_ft_mmd003_no_org_cls_knn_token_concat_pca50.h5ad"
ZS="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_zs_knn_fixed.h5ad"
ZS_ALL="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_zs_knn_fixed_token_concat_pca50.h5ad"
OUTPUT="${DATA_ROOT}/pandora_lung_filtered_93423_scib113_mmd003_no_org_cls_knn.csv"

cd "$REPO_ROOT"
test -s "$FT"
test -s "$FT_ALL"

srun "$PYTHON" scripts/benchmark_task3_scib113_precomputed.py \
  --source "$SOURCE" \
  --batch-key species \
  --label-key cell_type_ontology_term_id \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "scPRINT-2-FT-cell-token-mmd003-no-organism-classification-knn=${FT}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-FT-all-tokens-mmd003-no-organism-classification-knn=${FT_ALL}:token_concat_pca50" \
  --embedding "scPRINT-2-ZS-cell-token-knn=${ZS}:scprint_emb_cell_type_ontology_term_id" \
  --embedding "scPRINT-2-ZS-all-tokens-pca50=${ZS_ALL}:token_concat_pca50" \
  --output "$OUTPUT"
