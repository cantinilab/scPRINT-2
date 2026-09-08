#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-transcriptformer-%j.out
#SBATCH --error=slurm/pandora-transcriptformer-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
TF_ENV="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}"
INPUT="${PANDORA_INPUT:-${DATA_ROOT}/pandora_lung_human_one2one.h5ad}"
OUTPUT="${PANDORA_OUTPUT:-${DATA_ROOT}/pandora_lung_transcriptformer.h5ad}"
export PYTHONPATH="${SCRATCH_ROOT}/scprint_data/setuptools-overlay${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO_ROOT"
srun "${TF_ENV}/bin/python" scripts/run_task3_cross_species_embedding.py transcriptformer \
  --input "$INPUT" \
  --checkpoint "${WORK_ROOT}/models/transcriptformer/tf_metazoa" \
  --output "$OUTPUT" \
  --notebook-provenance notebooks/scPRINT-2-repro-notebooks/cross-species-embedding-pandora-lung.ipynb \
  --batch-key species \
  --cell-type-key cell_type
if [[ -z "${PANDORA_OUTPUT:-}" ]]; then
  touch "${DATA_ROOT}/TRANSCRIPTFORMER.COMPLETE"
fi
