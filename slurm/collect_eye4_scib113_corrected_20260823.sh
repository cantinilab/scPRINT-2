#!/bin/bash
#SBATCH --job-name=eye4-scib113-collect
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/scib113-collect-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/scib113-collect-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_corrected_20260823}"
BY_METHOD="${ROOT}/scib113/by_method"
OUTPUT="${ROOT}/scib113/eye4_scib113_corrected_fullmetrics_scores.csv"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

test ! -e "$OUTPUT"
test ! -e "${OUTPUT%.csv}.metadata.json"
test ! -e "${OUTPUT%.csv}.COMPLETE"

srun "$PYTHON" "${REPO_ROOT}/scripts/collect_eye4_scib113_scores.py" \
  --input "PCA=${BY_METHOD}/PCA/PCA.csv" \
  --input "Random-seed42=${BY_METHOD}/Random-seed42/Random-seed42.csv" \
  --input "scPRINT-2-ZS-native-cell-token=${BY_METHOD}/scPRINT-2-ZS-native-cell-token/scPRINT-2-ZS-native-cell-token.csv" \
  --input "scPRINT-2-FT-native-cell-token-MMD003=${BY_METHOD}/scPRINT-2-FT-native-cell-token-MMD003/scPRINT-2-FT-native-cell-token-MMD003.csv" \
  --input "TranscriptFormer-Metazoa-mouse-ortholog=${BY_METHOD}/TranscriptFormer-Metazoa-mouse-ortholog/TranscriptFormer-Metazoa-mouse-ortholog.csv" \
  --expected-method "PCA" \
  --expected-method "Random-seed42" \
  --expected-method "scPRINT-2-ZS-native-cell-token" \
  --expected-method "scPRINT-2-FT-native-cell-token-MMD003" \
  --expected-method "TranscriptFormer-Metazoa-mouse-ortholog" \
  --output "$OUTPUT"

test -s "$OUTPUT"
test -s "${OUTPUT%.csv}.metadata.json"
test -s "${OUTPUT%.csv}.COMPLETE"
