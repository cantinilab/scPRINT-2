#!/bin/bash
set -euo pipefail

# Run this lightweight network-only step on the Jean Zay submit node.  The
# Slurm preparation job consumes and validates the shared-filesystem outputs.
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_matched_20260823}"
mkdir -p "${ROOT}/geo"

download() {
  local accession="$1" filename="$2"
  local output="${ROOT}/geo/${filename}"
  local url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=${accession}&file=${filename}&format=file"
  if [[ ! -s "$output" ]]; then
    curl --fail --location --retry 12 --retry-all-errors --retry-delay 10 \
      --continue-at - --output "$output" "$url"
  fi
  gzip -t "$output"
  echo "VERIFIED_GEO ${accession} $(stat -c %s "$output") ${output}"
}

download GSE148371 GSE148371_Human_count_matrix.csv.gz
download GSE148373 GSE148373_MacaF_count_matrix.csv.gz
download GSE146186 GSE146186_Mouse_count_matrix.csv.gz
download GSE146187 GSE146187_Pig_count_matrix.csv.gz
sha256sum "${ROOT}/geo/"*.csv.gz > "${ROOT}/geo_sha256.tsv"
