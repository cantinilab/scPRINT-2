#!/bin/bash
set -euo pipefail

SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
OUTPUT_ROOT="${OP_COMMON_ROOT:-${SCRATCH_ROOT}/openproblems_common}"
BASE_URL="https://openproblems-data.s3.amazonaws.com/resources/datasets/cellxgene_census"

# Jean Zay's non-interactive submit-node shells return from .bashrc before its
# proxy exports. Respect explicit caller values, otherwise use the site proxy.
export http_proxy="${http_proxy:-http://prodprox.idris.fr:3128}"
export https_proxy="${https_proxy:-http://prodprox.idris.fr:3128}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

# These are the exact published OpenProblems log_cp10k common datasets.  They
# contain raw counts as well as the published normalized layer; op_scib.py
# deliberately recomputes normalization from counts instead of trusting that
# layer. Downloads must run on the Jean Zay submit node because compute nodes
# have no outbound network access.
if [[ $# -gt 0 ]]; then
  datasets=("$@")
else
  datasets=(
    dkd
    gtex_v9
    hypomap
    mouse_pancreas_atlas
  )
fi

for dataset in "${datasets[@]}"; do
  case "${dataset}" in
    dkd | gtex_v9 | hypomap | mouse_pancreas_atlas) ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      exit 2
      ;;
  esac
done

for dataset in "${datasets[@]}"; do
  destination_dir="${OUTPUT_ROOT}/cellxgene_census/${dataset}/log_cp10k"
  destination="${destination_dir}/dataset.h5ad"
  partial="${destination}.part"
  mkdir -p "${destination_dir}"
  if [[ -s "${destination}" ]]; then
    echo "Already present: ${destination}"
    continue
  fi
  echo "Downloading ${dataset} to ${destination}"
  curl -L --fail --retry 20 --retry-all-errors --retry-delay 5 \
    --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
    --continue-at - --progress-bar \
    "${BASE_URL}/${dataset}/log_cp10k/dataset.h5ad" \
    -o "${partial}"
  mv "${partial}" "${destination}"
done
