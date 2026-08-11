#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <main|transcriptformer> <notebook-path>" >&2
  exit 2
fi

ENV_KIND="$1"
NOTEBOOK="$2"
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${REPO_ROOT:-${WORK_ROOT}/scPRINT}"
export WORK="${WORK_ROOT}"
export SCRATCH="${SCRATCH_ROOT}"
export REPO_ROOT
export OP_SOLUTION_ROOT="${OP_SOLUTION_ROOT:-${SCRATCH_ROOT}/openproblems_reconstructed}"

# Some Jean Zay profile fragments probe optional unset variables. Keep strict
# mode for this script, but disable nounset only while the site profile loads.
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

case "${ENV_KIND}" in
  main)
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    ;;
  transcriptformer)
    # any_sub.sh loads the site CUDA 12.2 module. TranscriptFormer's pinned
    # PyTorch 2.5.1 wheel ships CUDA 12.4 libraries, so the site module would
    # shadow its matching nvJitLink and make torch fail before notebook startup.
    module unload cuda/12.2.0 || true
    TF_ENV="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}"
    PYTHON="${TF_ENV}/bin/python"
    export PYTHONPATH="${SCRATCH_ROOT}/scprint_data/setuptools-overlay${PYTHONPATH:+:${PYTHONPATH}}"
    export IPYTHONDIR="${SCRATCH_ROOT}/scprint_data/ipython"
    export PYTHONDONTWRITEBYTECODE=1
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ;;
  *)
    echo "unknown environment kind: ${ENV_KIND}" >&2
    exit 2
    ;;
esac

export PATH="$(dirname "${PYTHON}"):${PATH}"
cd "${REPO_ROOT}"
exec "${PYTHON}" -m papermill \
  --progress-bar \
  --autosave-cell-every 120 \
  -k python3 \
  --log-output \
  "${NOTEBOOK}" \
  "${NOTEBOOK}"
