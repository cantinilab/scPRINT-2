#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <dataset> <zeroshot|finetune> <output-dir>" >&2
  exit 2
fi

DATASET="$1"
MODE="$2"
OUTPUT_DIR="$3"
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${REPO_ROOT:-${WORK_ROOT}/scPRINT}"
SCRIPT_PATH="${SCPRINT2_OP_SCRIPT:-${REPO_ROOT}/scripts/scprint2_op_full_no_assay.py}"

set +u
source /etc/profile
set -u
module load r/4.4.1
export WORK="${WORK_ROOT}"
export SCRATCH="${SCRATCH_ROOT}"
export REPO_ROOT
export OP_SOLUTION_ROOT="${OP_SOLUTION_ROOT:-${SCRATCH_ROOT}/openproblems_reconstructed}"
export R_HOME="$(R RHOME)"
export R_LIBS_USER="${R_LIBS_USER:-${WORK_ROOT}/R/library/4.4}"
export SCIB_NATIVE_CACHE="${SCIB_NATIVE_CACHE:-${WORK_ROOT}/.cache/scib-native}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RUN_DIR="${OUTPUT_DIR}/work/${MODE}/${DATASET}"
mkdir -p "${RUN_DIR}/data"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${RUN_DIR}"
EXTRA_ARGS=()
if [[ "${CLASSIFICATION_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--classification-only)
fi
exec "${REPO_ROOT}/.venv/bin/python" "${SCRIPT_PATH}" \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --input "${REPO_ROOT}/data/temp/cellxgene_census/${DATASET}_proc.h5ad" \
  --checkpoint "${WORK_ROOT}/models/small-v2.ckpt" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
