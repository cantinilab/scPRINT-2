#!/bin/bash
set -euo pipefail
set +u
source /etc/profile
module load python/3.9.12
module load r/4.4.1
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
ENV_ROOT="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}"
UV="${UV_BIN:-/linkhome/rech/gennth01/${USER}/.local/bin/uv}"
export R_HOME="$(R RHOME)"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  "$UV" venv --python "$(command -v python3)" "$ENV_ROOT"
fi

"$UV" pip install --python "${ENV_ROOT}/bin/python" \
  "setuptools==59.8.0" \
  "numpy==1.21.6" \
  "pandas==1.3.5" \
  "scipy==1.7.3" \
  "scikit-learn==1.0.2" \
  "h5py==3.7.0" \
  "matplotlib==3.5.2" \
  "seaborn==0.11.2" \
  "numba==0.55.2" \
  "llvmlite==0.38.1" \
  "pynndescent==0.5.7" \
  "umap-learn==0.5.3" \
  "anndata==0.8.0" \
  "scanpy==1.9.1" \
  "scib==1.1.3" \
  "rpy2==3.4.2" \
  "anndata2ri==1.0.6" \
  "igraph==0.10.3" \
  "leidenalg==0.9.1" \
  "louvain==0.8.0"

R_LIB="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}"
mkdir -p "$R_LIB"
COMMON_R_LIB="${WORK_ROOT}/R/library/4.4"
export R_LIBS_USER="${R_LIB}:${COMMON_R_LIB}"
R_ARCHIVES="${PAPER_SCIB_R_ARCHIVES:-${WORK_ROOT}/.cache/task3-paper-scib-1.1.3/r-src}"
archives=(
  Rcpp_1.1.2.tar.gz
  RcppArmadillo_15.4.2-1.tar.gz
  RANN_2.6.2.tar.gz
  data.table_1.14.2.tar.gz
  NbClust_3.0.tar.gz
  mclust_6.0.0.tar.gz
  lisi-master.tar.gz
)
for archive in "${archives[@]}"; do
  if [[ ! -s "${R_ARCHIVES}/${archive}" ]]; then
    echo "Missing staged R archive: ${R_ARCHIVES}/${archive}" >&2
    exit 4
  fi
  if [[ "$archive" == "lisi-master.tar.gz" ]]; then
    LISI_BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$LISI_BUILD_DIR"' EXIT
    tar --extract --gzip --file="${R_ARCHIVES}/${archive}" --directory="$LISI_BUILD_DIR"
    sed -i 's/CXX_STD = CXX11/CXX_STD = CXX14/' \
      "$LISI_BUILD_DIR"/LISI-master/src/Makevars \
      "$LISI_BUILD_DIR"/LISI-master/src/Makevars.win
    R CMD INSTALL --library="$R_LIB" "$LISI_BUILD_DIR/LISI-master"
    rm -rf "$LISI_BUILD_DIR"
    trap - EXIT
  else
    R CMD INSTALL --library="$R_LIB" "${R_ARCHIVES}/${archive}"
  fi
done

"${ENV_ROOT}/bin/python" - <<'PY'
import importlib.metadata as metadata
import pkg_resources
import numba
import scanpy
import scib

assert metadata.version("scib") == "1.1.3"
assert metadata.version("scanpy") == "1.9.1"
assert metadata.version("setuptools") == "59.8.0"
print({name: metadata.version(name) for name in ("scib", "scanpy", "anndata", "numpy", "pandas", "scikit-learn", "setuptools")})
PY
Rscript -e 'for (p in c("data.table", "NbClust", "mclust", "lisi", "kBET")) stopifnot(requireNamespace(p, quietly=TRUE)); sessionInfo()'
