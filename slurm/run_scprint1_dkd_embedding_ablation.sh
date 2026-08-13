#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <embed|score> <output-dir> [variant]" >&2
  exit 2
fi

MODE="$1"
OUTPUT_DIR="$2"
VARIANT="${3:-}"
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${REPO_ROOT:-${WORK_ROOT}/scPRINT}"
SCRIPT_PATH="${SCPRINT1_ABLATION_SCRIPT:-${REPO_ROOT}/scripts/scprint1_dkd_embedding_ablation.py}"

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

cd "${REPO_ROOT}"
case "${MODE}" in
  embed)
    exec .venv/bin/python "${SCRIPT_PATH}" embed \
      --input "${REPO_ROOT}/data/temp/cellxgene_census/dkd_proc.h5ad" \
      --checkpoint "${WORK_ROOT}/models/ogvvg2z7-v1.ckpt" \
      --output-dir "${OUTPUT_DIR}" \
      ${SCPRINT1_ABLATION_EMBED_ARGS:-}
    ;;
  score)
    if [[ -z "${VARIANT}" ]]; then
      echo "score mode requires a variant" >&2
      exit 2
    fi
    exec .venv/bin/python "${SCRIPT_PATH}" score \
      --input "${REPO_ROOT}/data/temp/cellxgene_census/dkd_proc.h5ad" \
      --output-dir "${OUTPUT_DIR}" \
      --variant "${VARIANT}"
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
