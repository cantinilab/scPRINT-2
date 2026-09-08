#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-download-%j.out
#SBATCH --error=slurm/pandora-download-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
MANIFEST="${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/pandora_lung_atlas_manifest.tsv"
mkdir -p "${DATA_ROOT}/rds"

download_one() {
  local species="$1" organism="$2" filename="$3" url="$4"
  local output="${DATA_ROOT}/rds/${filename}"
  local expected actual
  expected=$(curl --fail --silent --show-error --head --location "$url" \
    | tr -d '\r' | awk 'tolower($1) == "content-length:" {print $2}' | tail -1)
  actual=$(stat -c %s "$output" 2>/dev/null || echo 0)
  if [[ "$actual" != "$expected" ]]; then
    curl --fail --location --retry 12 --retry-all-errors --retry-delay 10 \
      --continue-at - --output "$output" "$url"
  fi
  actual=$(stat -c %s "$output")
  [[ "$actual" == "$expected" ]] || { echo "Size mismatch for ${species}" >&2; return 1; }
  echo "DOWNLOADED ${species} $(stat -c %s "$output") ${output}"
}
export -f download_one
export DATA_ROOT
tail -n +2 "$MANIFEST" | xargs -P 6 -n 4 bash -c 'download_one "$@"' _
touch "${DATA_ROOT}/DOWNLOAD.COMPLETE"
