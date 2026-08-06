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
R_ENV="${OP_SCIB_R_ENV:-${WORK_ROOT}/venvs/op-scib-r}"
TF_ENV="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}"
UV="${UV:-${HOME}/.local/bin/uv}"

export R_HOME="${R_ENV}/lib/R"
export PATH="${R_ENV}/bin:${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORK_ROOT}/.cache/uv}"
export SCIB_NATIVE_CACHE="${SCIB_NATIVE_CACHE:-${WORK_ROOT}/.cache/scib-native}"

mkdir -p "${WORK_ROOT}/venvs" "${UV_CACHE_DIR}" "${SCIB_NATIVE_CACHE}"

if [[ ! -x "${R_ENV}/bin/R" ]]; then
  /gpfslocalsup/pub/anaconda-py3/2023.09/bin/conda create -y \
    -p "${R_ENV}" -c conda-forge \
    r-base=4.3 r-remotes r-fnn r-matrix
fi

"${R_ENV}/bin/Rscript" -e \
  'if (!requireNamespace("kBET", quietly=TRUE)) remotes::install_github("theislab/kBET", ref="0.99.6")'

cd "${REPO_ROOT}"

"${UV}" pip install --python .venv/bin/python \
  'jax[cuda12]==0.7.0' rpy2 'anndata2ri==1.3.1'

"${UV}" pip install --python "${TF_ENV}/bin/python" \
  'scib==1.1.7' 'jax[cuda12]==0.10.2' rpy2 'anndata2ri==1.3.1'

R_HOME="${R_HOME}" .venv/bin/python -c \
  'from op_scib import prepare_op_scib_environment; print(prepare_op_scib_environment())'

R_HOME="${R_HOME}" "${TF_ENV}/bin/python" -c \
  'from op_scib import prepare_op_scib_environment; print(prepare_op_scib_environment())'
