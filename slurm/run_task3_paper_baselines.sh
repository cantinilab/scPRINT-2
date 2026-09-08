#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-paper-baselines-%j.out
#SBATCH --error=slurm/task3-paper-baselines-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
OUTPUT="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3/task3_paper_baselines.h5ad"

"$PYTHON" "$REPO_ROOT/scripts/prepare_task3_paper_baselines.py" \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --seed 42 \
  --n-components 50
