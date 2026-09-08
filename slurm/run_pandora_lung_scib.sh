#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scib-%j.out
#SBATCH --error=slurm/pandora-scib-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
cd "$REPO_ROOT"
srun .venv/bin/python scripts/benchmark_pandora_lung_scib.py \
  --expression "${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad" \
  --embedding "${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2.h5ad" \
  --embedding-name scprint2_small_v2 \
  --additional-embedding "scprint2_small_v2_species_mmd=${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_species_mmd.h5ad:scprint_emb" \
  --output "${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_species_mmd_scib_all_methods.csv" \
  --embedding-key scprint_emb \
  --batch-key species \
  --label-key cell_type_ontology_term_id \
  --exclude-label unknown \
  --exclude-label unmapped \
  --exclude-label mix \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
