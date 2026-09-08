#!/bin/bash
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=slurm/task3-fresh-umaps-%j.out
#SBATCH --error=slurm/task3-fresh-umaps-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
RESULT_ROOT="${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_umaps_scoring_graph_20260818"
ONTOLOGY_PARQUET="${RESULT_ROOT}/cell_ontology_names_2025-12-17.json"

test ! -e "${RESULT_ROOT}/COMPLETE"
test -s "${ONTOLOGY_PARQUET}"
mkdir -p "${RESULT_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun "${PYTHON}" "${REPO_ROOT}/scripts/plot_task3_fresh_umaps.py" \
  --scprint2 "${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_ontology_knn_mmd003_20260815/scprint2_zero_shot_small_v2_knn.h5ad" \
  --scprint2-ft "${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_ontology_knn_mmd003_20260815/scprint2_ft_small_v2_mmd003_knn.h5ad" \
  --transcriptformer "${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_transcriptformer_metazoa_scib113_20260817/transcriptformer_metazoa_fresh.h5ad" \
  --ontology-parquet "${ONTOLOGY_PARQUET}" \
  --output-dir "${RESULT_ROOT}/plots"
