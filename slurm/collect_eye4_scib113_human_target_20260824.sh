#!/bin/bash
#SBATCH --job-name=eye4-human-collect
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/scib113-collect-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/scib113-collect-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_human_target_hybrid_20260824}"
BY_METHOD="${ROOT}/scib113/by_method"
OUTPUT="${ROOT}/scib113/eye4_scib113_human_target_fullmetrics_scores.csv"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

METHODS=(
  "PCA"
  "Random-seed42"
  "scPRINT-2-ZS-human-target-cell-token"
  "scPRINT-2-FT-human-target-cell-token-MMD003"
  "TranscriptFormer-Metazoa-human-target"
)

test ! -e "$OUTPUT"
test ! -e "${OUTPUT%.csv}.metadata.json"
test ! -e "${OUTPUT%.csv}.COMPLETE"

ARGS=()
for method in "${METHODS[@]}"; do
  ARGS+=(--input "${method}=${BY_METHOD}/${method}/${method}.csv")
  ARGS+=(--expected-method "$method")
done

srun "$PYTHON" "${REPO_ROOT}/scripts/collect_eye4_scib113_scores.py" \
  "${ARGS[@]}" \
  --output "$OUTPUT"

test -s "$OUTPUT"
test -s "${OUTPUT%.csv}.metadata.json"
test -s "${OUTPUT%.csv}.COMPLETE"
