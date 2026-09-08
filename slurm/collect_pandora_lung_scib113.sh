#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/pandora-scib113-collect-%j.out
#SBATCH --error=slurm/pandora-scib113-collect-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"

cd "$REPO_ROOT"
srun "$PYTHON" scripts/collect_pandora_scib113.py \
  --data-root "$DATA_ROOT" \
  --modern "${DATA_ROOT}/pandora_lung_filtered_93423_scib_mmd003_no_org_cls_knn.tsv" \
  --output "${DATA_ROOT}/pandora_lung_filtered_93423_scib113_five_methods.tsv"
