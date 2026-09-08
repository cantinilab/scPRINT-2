#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scprint2-ft-%j.out
#SBATCH --error=slurm/pandora-scprint2-ft-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
INPUT="${PANDORA_INPUT:-${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad}"
OUTPUT="${PANDORA_OUTPUT:-${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_species_mmd.h5ad}"
cd "$REPO_ROOT"
srun .venv/bin/python scripts/run_task3_cross_species_embedding.py scprint2-ft \
  --input "$INPUT" \
  --checkpoint "${WORK_ROOT}/models/small-v2.ckpt" \
  --output "$OUTPUT" \
  --notebook-provenance notebooks/scPRINT-2-repro-notebooks/cross-species-embedding-pandora-lung.ipynb \
  --align-checkpoint-genes \
  --batch-key species \
  --cell-type-key cell_type_ontology_term_id \
  --embed-how "random expr" \
  --embed-max-len 2200 \
  --finetune-max-len 2200 \
  --mmd-target species \
  --exclude-train-label unknown \
  --exclude-train-label unmapped \
  --exclude-train-label mix \
  --num-epochs "${PANDORA_NUM_EPOCHS:-8}"
if [[ -z "${PANDORA_OUTPUT:-}" ]]; then
  touch "${DATA_ROOT}/SCPRINT2_SMALL_V2_SPECIES_MMD_MOUSE_SPACE_FILTERED_93423.COMPLETE"
fi
