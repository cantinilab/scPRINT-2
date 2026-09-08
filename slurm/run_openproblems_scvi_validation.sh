#!/bin/bash
set -euo pipefail

set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${REPO_ROOT:-${WORK_ROOT}/scPRINT}"
DATASET="${1:-dkd}"
DATASET_ID="cellxgene_census/${DATASET}"
INPUT="${WORK_ROOT}/openproblems_common/${DATASET_ID}/log_cp10k/dataset.h5ad"
OUTPUT_ROOT="${2:-${SCRATCH_ROOT}/scprint_data/openproblems_scvi_validation/${DATASET}}"
CONTAINER="${OPENPROBLEMS_SCVI_CONTAINER:-${SCRATCH_ROOT}/containers/openproblems_base_pytorch_nvidia_1.sif}"
EMBEDDING="${OUTPUT_ROOT}/${DATASET}_scvi_openproblems_embedding.h5ad"
SCORES="${OUTPUT_ROOT}/${DATASET}_scvi_openproblems_op_scib.csv"
mkdir -p "${OUTPUT_ROOT}" "${SCRATCH_ROOT}/.cache/singularity-openproblems"

export SINGULARITY_CACHEDIR="${SCRATCH_ROOT}/.cache/singularity-openproblems"
export OP_SOLUTION_ROOT="${OP_SOLUTION_ROOT:-${SCRATCH_ROOT}/openproblems_reconstructed}"
export R_HOME="$(R RHOME)"
export R_LIBS_USER="${R_LIBS_USER:-${WORK_ROOT}/R/library/4.4}"
export SCIB_NATIVE_CACHE="${SCIB_NATIVE_CACHE:-${WORK_ROOT}/.cache/scib-native}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
SINGULARITY_BIN="${SINGULARITY_BIN:-/gpfslocalsys/singularity/singularity-3.8.5/bin/singularity}"
SCVI_ARGS=(--input "${INPUT}" --output "${EMBEDDING}")
if [[ -n "${SCVI_SEED:-}" ]]; then
  SCVI_ARGS+=(--seed "${SCVI_SEED}")
fi

if [[ -n "${SCVI_OVERLAY:-}" ]]; then
  PYTHONPATH="${SCVI_OVERLAY}:${PYTHONPATH}" \
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/run_openproblems_scvi.py" \
      "${SCVI_ARGS[@]}"
else
  "${SINGULARITY_BIN}" exec --nv --bind /lustre:/lustre "${CONTAINER}" \
    python "${REPO_ROOT}/scripts/run_openproblems_scvi.py" \
      "${SCVI_ARGS[@]}"
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/score_openproblems_embedding.py" \
  --input "${EMBEDDING}" \
  --output "${SCORES}" \
  --dataset "${DATASET_ID}"
