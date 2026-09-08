#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scprint1-%j.out
#SBATCH --error=slurm/pandora-scprint1-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
INPUT="${PANDORA_INPUT:-${DATA_ROOT}/pandora_lung_human_one2one.h5ad}"
OUTPUT="${PANDORA_OUTPUT:-${DATA_ROOT}/pandora_lung_scprint1.h5ad}"

cd "$REPO_ROOT"
srun .venv/bin/python scripts/run_task3_cross_species_embedding.py scprint-zero-shot \
  --input "$INPUT" \
  --checkpoint "${WORK_ROOT}/models/ogvvg2z7-v1.ckpt" \
  --output "$OUTPUT" \
  --notebook-provenance notebooks/scPRINT-2-repro-notebooks/cross-species-embedding-pandora-lung.ipynb \
  --align-checkpoint-genes \
  --batch-key species \
  --cell-type-key cell_type
touch "${DATA_ROOT}/SCPRINT1.COMPLETE"
