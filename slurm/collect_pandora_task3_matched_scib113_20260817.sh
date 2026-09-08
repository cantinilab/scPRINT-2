#!/bin/bash
#SBATCH --partition=prepost
#SBATCH --account=wbg@v100
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm/pandora-task3-matched-collect-%j.out
#SBATCH --error=slurm/pandora-task3-matched-collect-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RUN_ID="pandora_task3_matched_20260817"
ROOT="${WORK_ROOT}/data/pandora_lung/${RUN_ID}"
OUTPUT="${ROOT}/pandora_task3_matched_scib113_scores.tsv"

cd "$REPO_ROOT"
srun "$PYTHON" scripts/collect_pandora_task3_matched_scib113.py \
  --score-dir "${ROOT}/scores" \
  --output "$OUTPUT"
