#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-token-concat-%j.out
#SBATCH --error=slurm/task3-token-concat-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3"
OUT="${ROOT}/task3_token_concat_pca50"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$PYTHON" "$REPO_ROOT/scripts/build_task3_token_concat_pca.py" \
  --method scPRINT-1 \
  --input "${ROOT}/scprint1_embeddings.h5ad" \
  --output "${OUT}/scprint1_token_concat_pca50.h5ad"
"$PYTHON" "$REPO_ROOT/scripts/build_task3_token_concat_pca.py" \
  --method scPRINT-2 \
  --input "${ROOT}/scprint_zero_shot_embeddings.h5ad" \
  --output "${OUT}/scprint2_token_concat_pca50.h5ad"
"$PYTHON" "$REPO_ROOT/scripts/build_task3_token_concat_pca.py" \
  --method scPRINT-2-FT \
  --input "${ROOT}/scprint_mmd_embeddings.h5ad" \
  --output "${OUT}/scprint2_ft_token_concat_pca50.h5ad"
