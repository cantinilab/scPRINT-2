#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <dataset-slug> <common-dataset.h5ad> <output-root> [--reference-only]" >&2
  exit 2
fi

DATASET="$1"
INPUT_COMMON="$2"
OUTPUT_ROOT="$3"
MODE="${4:-}"
EXTRA_ARGS=()
if [[ -n "${MODE}" && "${MODE}" != "--reference-only" ]]; then
  echo "unknown mode: ${MODE}" >&2
  exit 2
fi
if [[ -n "${MODE}" ]]; then
  EXTRA_ARGS+=("${MODE}")
fi
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${REPO_ROOT:-${WORK_ROOT}/scPRINT}"

set +u
source /etc/profile
set -u
module load r/4.4.1

export R_HOME="$(R RHOME)"
export R_LIBS_USER="${R_LIBS_USER:-${WORK_ROOT}/R/library/4.4}"
export SCIB_NATIVE_CACHE="${SCIB_NATIVE_CACHE:-${WORK_ROOT}/.cache/scib-native}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${REPO_ROOT}"
exec .venv/bin/python scripts/validate_op_no_integration.py \
  --input-common "${INPUT_COMMON}" \
  --dataset "${DATASET}" \
  --output-root "${OUTPUT_ROOT}" \
  --silhouette-backend jax \
  "${EXTRA_ARGS[@]}"
