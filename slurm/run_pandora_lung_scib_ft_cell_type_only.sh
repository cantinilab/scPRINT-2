#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scib-ft-ct-only-%j.out
#SBATCH --error=slurm/pandora-scib-ft-ct-only-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
FT_CT="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_ft_cell_type_only.h5ad"
ZERO="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2.h5ad"
FT_MMD="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_species_mmd.h5ad"
OUTPUT="${DATA_ROOT}/pandora_lung_filtered_93423_scib_ft_cell_type_only.csv"

cd "$REPO_ROOT"
srun .venv/bin/python scripts/benchmark_pandora_lung_scib.py \
  --expression "${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad" \
  --embedding "$FT_CT" \
  --embedding-key scprint_emb_cell_type_ontology_term_id \
  --embedding-name scprint2_small_v2_ft_cell_type_only \
  --additional-embedding "scprint2_small_v2_zero_shot_cell_type=${ZERO}:scprint_emb_cell_type_ontology_term_id" \
  --additional-embedding "scprint2_small_v2_species_mmd_cell_type=${FT_MMD}:scprint_emb_cell_type_ontology_term_id" \
  --output "$OUTPUT" \
  --batch-key species \
  --label-key cell_type_ontology_term_id \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
