#!/bin/bash
#SBATCH --job-name=eye4-biomart
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/biomart-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/biomart-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

export http_proxy="${http_proxy:-http://prodprox.idris.fr:3128}"
export HTTP_PROXY="$http_proxy"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTPS_PROXY="$https_proxy"

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_corrected_20260823}"
OUTPUT="${ROOT}/orthologs/biomart"
mkdir -p "$OUTPUT"
test ! -e "${OUTPUT}/BIOMART.COMPLETE"
srun "$PYTHON" "${REPO_ROOT}/scripts/fetch_eye4_biomart_orthologs.py" \
  --output-dir "$OUTPUT"
test -s "${OUTPUT}/BIOMART.COMPLETE"
