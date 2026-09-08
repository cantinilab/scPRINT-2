#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=slurm/task3-paper-scvi-%j.out
#SBATCH --error=slurm/task3-paper-scvi-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${PAPER_SCVI_ENV:-${SCRATCH_ROOT}/venvs/task3-paper-scvi-0.19.0}/bin/python"
INPUT="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
OUTPUT="$1"
SEED="${2:-42}"

cd "$REPO_ROOT"
srun "$PYTHON" scripts/run_task3_paper_scvi.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --seed "$SEED"
