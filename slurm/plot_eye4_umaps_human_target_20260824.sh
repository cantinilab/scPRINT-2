#!/bin/bash
#SBATCH --job-name=eye4-human-umap
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/umaps-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/umaps-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_human_target_hybrid_20260824}"
GRAPHS="${ROOT}/scib113/precomputed_graphs"
OUTPUT="${ROOT}/figures/scored_umaps"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

test -s "${ROOT}/scib113/eye4_scib113_human_target_fullmetrics_scores.COMPLETE"
srun "$PYTHON" "${REPO_ROOT}/scripts/plot_eye4_scored_umaps.py" \
  --graph "PCA=${GRAPHS}/PCA.h5ad" \
  --graph "Random=${GRAPHS}/Random-seed42.h5ad" \
  --graph "scPRINT-2 ZS=${GRAPHS}/scPRINT-2-ZS-human-target-cell-token.h5ad" \
  --graph "scPRINT-2 FT=${GRAPHS}/scPRINT-2-FT-human-target-cell-token-MMD003.h5ad" \
  --graph "TranscriptFormer=${GRAPHS}/TranscriptFormer-Metazoa-human-target.h5ad" \
  --output-dir "$OUTPUT" \
  --random-state 42 \
  --min-dist 0.5 \
  --spread 1.0

test -s "${OUTPUT}/eye4_scored_umaps_overview.png"
test -s "${OUTPUT}/eye4_scored_umaps.metadata.json"
test -s "${OUTPUT}/UMAPS.COMPLETE"
