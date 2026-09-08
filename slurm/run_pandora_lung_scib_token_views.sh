#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scib-token-views-%j.out
#SBATCH --error=slurm/pandora-scib-token-views-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
ZERO="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2.h5ad"
FT="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_species_mmd.h5ad"
ZERO_CONCAT="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_token_concat_pca50.h5ad"
FT_CONCAT="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_species_mmd_token_concat_pca50.h5ad"
OUTPUT="${DATA_ROOT}/pandora_lung_filtered_93423_scib_token_views.csv"

cd "$REPO_ROOT"
srun .venv/bin/python scripts/build_task3_token_concat_pca.py \
  --input "$ZERO" --output "$ZERO_CONCAT" --method scprint2_small_v2_token_concat_pca50
srun .venv/bin/python scripts/build_task3_token_concat_pca.py \
  --input "$FT" --output "$FT_CONCAT" --method scprint2_small_v2_species_mmd_token_concat_pca50
srun .venv/bin/python scripts/benchmark_pandora_lung_scib.py \
  --expression "${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad" \
  --embedding "$ZERO" \
  --embedding-key scprint_emb_cell_type_ontology_term_id \
  --embedding-name scprint2_small_v2_cell_type \
  --additional-embedding "scprint2_small_v2_species_mmd_cell_type=${FT}:scprint_emb_cell_type_ontology_term_id" \
  --additional-embedding "scprint2_small_v2_token_concat_pca50=${ZERO_CONCAT}:token_concat_pca50" \
  --additional-embedding "scprint2_small_v2_species_mmd_token_concat_pca50=${FT_CONCAT}:token_concat_pca50" \
  --output "$OUTPUT" \
  --batch-key species \
  --label-key cell_type_ontology_term_id \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
