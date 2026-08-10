#!/bin/bash
set -euo pipefail

SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
OUTPUT_ROOT="${OP_COMMON_ROOT:-${SCRATCH_ROOT}/openproblems_common}"
BASE_URL="https://openproblems-data.s3.amazonaws.com/resources/datasets/cellxgene_census"

# These are the exact published OpenProblems log_cp10k common datasets.  They
# contain raw counts as well as the published normalized layer; op_scib.py
# deliberately recomputes normalization from counts instead of trusting that
# layer. Downloads must run on the Jean Zay submit node because compute nodes
# have no outbound network access.
datasets=(
  dkd
  gtex_v9
  hypomap
  mouse_pancreas_atlas
)

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
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
    "${BASE_URL}/${dataset}/log_cp10k/dataset.h5ad" \
    -o "${partial}"
  mv "${partial}" "${destination}"
done
