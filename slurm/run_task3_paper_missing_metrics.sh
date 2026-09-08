#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-paper-missing-metrics-%j.out
#SBATCH --error=slurm/task3-paper-missing-metrics-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

METHOD="$1"
PAPER_ROOT="$2"
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
export R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

test -s "${PAPER_ROOT}/output/evaluation/saturn/task3_saturn_ARI.txt"
test ! -e "${PAPER_ROOT}/task3_missing_metrics.csv"
test ! -e "${PAPER_ROOT}/task3_missing_metrics.COMPLETE"

srun "$PYTHON" "$REPO_ROOT/scripts/run_task3_paper_missing_metrics.py" \
  --paper-root "$PAPER_ROOT" \
  --method "$METHOD"
