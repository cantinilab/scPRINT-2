#!/bin/bash
set -euo pipefail
set +u
source /etc/profile
module load python/3.9.12
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
ENV_ROOT="${PAPER_SCVI_ENV:-${SCRATCH_ROOT}/venvs/task3-paper-scvi-0.19.0}"
UV="${UV_BIN:-/linkhome/rech/gennth01/${USER}/.local/bin/uv}"
export UV_CACHE_DIR="${PAPER_SCVI_UV_CACHE:-${SCRATCH_ROOT}/.cache/uv-task3-paper-scvi}"
mkdir -p "$UV_CACHE_DIR"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  "$UV" venv --python "$(command -v python3)" "$ENV_ROOT"
fi

"$UV" pip install --python "${ENV_ROOT}/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu102 \
  --index-strategy unsafe-best-match \
  "setuptools==59.8.0" \
  "numpy==1.21.6" \
  "pandas==1.3.5" \
  "scipy==1.7.3" \
  "scikit-learn==1.0.2" \
  "scikit-misc==0.1.4" \
  "h5py==3.7.0" \
  "matplotlib==3.5.2" \
  "numba==0.55.2" \
  "llvmlite==0.38.1" \
  "pynndescent==0.5.7" \
  "umap-learn==0.5.3" \
  "anndata==0.8.0" \
  "scanpy==1.9.1" \
  "torch==1.12.1+cu102" \
  "torchvision==0.13.1+cu102" \
  "jax==0.3.25" \
  "jaxlib @ https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.3.25-cp39-cp39-manylinux2014_x86_64.whl" \
  "chex==0.1.5" \
  "flax==0.6.4" \
  "ml-collections==0.1.1" \
  "mudata==0.2.1" \
  "numpyro==0.10.1" \
  "optax==0.1.4" \
  "orbax==0.1.0" \
  "pyro-ppl==1.8.4" \
  "pytorch-lightning==1.7.7" \
  "torchmetrics==0.11.1" \
  "tensorstore==0.1.28" \
  "scvi-tools==0.19.0"

"${ENV_ROOT}/bin/python" - <<'PY'
import importlib.metadata as metadata
import pkg_resources
import scanpy
import scvi
import torch

expected = {
    "scvi-tools": "0.19.0",
    "scanpy": "1.9.1",
    "anndata": "0.8.0",
    "numpy": "1.21.6",
    "pandas": "1.3.5",
    "scikit-learn": "1.0.2",
    "torch": "1.12.1+cu102",
    "jax": "0.3.25",
    "jaxlib": "0.3.25",
    "setuptools": "59.8.0",
}
versions = {name: metadata.version(name) for name in expected}
assert versions == expected, (versions, expected)
print(versions)
print({"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda})
PY
