#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scprint2-knn-%j.out
#SBATCH --error=slurm/pandora-scprint2-knn-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
INPUT="${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad"
OUTPUT="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_knn.h5ad"

cd "$REPO_ROOT"
srun .venv/bin/python scripts/run_task3_cross_species_embedding.py scprint2-zero-shot \
  --input "$INPUT" \
  --checkpoint "${WORK_ROOT}/models/small-v2.ckpt" \
  --output "$OUTPUT" \
  --notebook-provenance notebooks/scPRINT-2-repro-notebooks/cross-species-embedding-pandora-lung.ipynb \
  --align-checkpoint-genes \
  --use-knn \
  --batch-key species \
  --cell-type-key cell_type_ontology_term_id \
  --embed-how "random expr" \
  --embed-max-len 2200

touch "${DATA_ROOT}/SCPRINT2_SMALL_V2_KNN_MOUSE_SPACE_FILTERED_93423.COMPLETE"
