#!/bin/bash
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm/task3-pca-umap-%j.out
#SBATCH --error=slurm/task3-pca-umap-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
BASELINE="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3/task3_paper_baselines.h5ad"
REFERENCE="${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_umaps_scoring_graph_20260818/plots/task3_fresh_scored_embeddings_umaps.h5ad"
OUTPUT_ROOT="${REPO_ROOT}/data/results/cross_species_embedding/task3_pca_umap_scoring_graph_20260821"
OUTPUT="${OUTPUT_ROOT}/task3_pca_umap_scoring_graph.h5ad"

test -s "${BASELINE}"
test -s "${REFERENCE}"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun "${PYTHON}" "${REPO_ROOT}/scripts/build_task3_pca_umap.py" \
  --pca-baseline "${BASELINE}" \
  --reference-coordinates "${REFERENCE}" \
  --output "${OUTPUT}"
