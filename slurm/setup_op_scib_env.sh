#!/bin/bash
set -euo pipefail

if [[ -f /etc/profile.d/proxy.sh ]]; then
  source /etc/profile.d/proxy.sh
elif [[ -z "${https_proxy:-}" ]]; then
  export http_proxy="http://prodprox.idris.fr:3128"
  export https_proxy="${http_proxy}"
  export HTTP_PROXY="${http_proxy}"
  export HTTPS_PROXY="${https_proxy}"
fi

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${REPO_ROOT:-${WORK_ROOT}/scPRINT}"
TF_ENV="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}"
UV="${UV:-${HOME}/.local/bin/uv}"

if ! command -v R >/dev/null 2>&1; then
  echo "R is unavailable. Run 'module load r/4.4.1' on the submit node first." >&2
  exit 1
fi

export R_HOME="${R_HOME:-$(R RHOME)}"
export R_LIBS_USER="${R_LIBS_USER:-${WORK_ROOT}/R/library/4.4}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORK_ROOT}/.cache/uv}"
export SCIB_NATIVE_CACHE="${SCIB_NATIVE_CACHE:-${WORK_ROOT}/.cache/scib-native}"

mkdir -p "${R_LIBS_USER}" "${UV_CACHE_DIR}" "${SCIB_NATIVE_CACHE}"

Rscript -e \
  'if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org"); if (!requireNamespace("kBET", quietly=TRUE)) remotes::install_url("https://codeload.github.com/theislab/kBET/tar.gz/refs/heads/master", dependencies=TRUE)'

cd "${REPO_ROOT}"

"${UV}" pip install --python .venv/bin/python \
  'jax[cuda12]==0.7.0' rpy2 'anndata2ri==1.3.1'

"${UV}" pip install --python "${TF_ENV}/bin/python" \
  'scib==1.1.7' 'jax[cuda12]==0.10.2' rpy2 'anndata2ri==1.3.1'

R_HOME="${R_HOME}" .venv/bin/python -c \
  'from op_scib import prepare_op_scib_environment; print(prepare_op_scib_environment())'

R_HOME="${R_HOME}" "${TF_ENV}/bin/python" -c \
  'from op_scib import prepare_op_scib_environment; print(prepare_op_scib_environment())'
