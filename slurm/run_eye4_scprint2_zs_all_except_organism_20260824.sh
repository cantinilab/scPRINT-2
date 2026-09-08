#!/bin/bash
#SBATCH --job-name=eye4-zs-all-no-org
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-task=32
#SBATCH --time=01:15:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/zs-all-no-org-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/zs-all-no-org-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
SCIB_PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_human_target_hybrid_20260824}"
SOURCE="${ROOT}/eye4_human_ensembl_small_v2.h5ad"
ZERO_SHOT="${ROOT}/embeddings/scprint2_zero_shot_small_v2_human_target_knn.h5ad"
CONCAT="${ROOT}/embeddings/scprint2_zero_shot_small_v2_human_target_all_tokens_except_organism_pca50.h5ad"
METHOD="scPRINT-2-ZS-human-target-all-tokens-except-organism-PCA50"
OLD_SCIB="${ROOT}/scib113"
NEW_SCIB="${ROOT}/scib113_all_tokens_except_organism"
METHOD_ROOT="${NEW_SCIB}/by_method/${METHOD}"
METHOD_SCORE="${METHOD_ROOT}/${METHOD}.csv"
COMBINED="${NEW_SCIB}/eye4_scib113_human_target_six_methods_scores.csv"
FIGURES="${ROOT}/figures/scored_umaps_all_tokens_except_organism"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

test -s "$SOURCE"
test -s "$ZERO_SHOT"
test -s "${ZERO_SHOT%.h5ad}.metadata.json"
test ! -e "$CONCAT"
test ! -e "${CONCAT%.h5ad}.metadata.json"
test ! -e "$NEW_SCIB"
test ! -e "$FIGURES"
mkdir -p "$METHOD_ROOT"

srun "$PYTHON" "${REPO_ROOT}/scripts/build_task3_token_concat_pca.py" \
  --input "$ZERO_SHOT" \
  --output "$CONCAT" \
  --method "$METHOD" \
  --token-policy all-except-organism \
  --seed 42

srun "$SCIB_PYTHON" "${REPO_ROOT}/scripts/benchmark_task3_scib113_precomputed.py" \
  --source "$SOURCE" \
  --batch-key species \
  --label-key celltype \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "${METHOD}=${CONCAT}:token_concat_pca50" \
  --output "$METHOD_SCORE"

METHODS=(
  "PCA"
  "Random-seed42"
  "scPRINT-2-ZS-human-target-cell-token"
  "scPRINT-2-FT-human-target-cell-token-MMD003"
  "TranscriptFormer-Metazoa-human-target"
)
COLLECT_ARGS=()
for method in "${METHODS[@]}"; do
  COLLECT_ARGS+=(--input "${method}=${OLD_SCIB}/by_method/${method}/${method}.csv")
  COLLECT_ARGS+=(--expected-method "$method")
done
COLLECT_ARGS+=(--input "${METHOD}=${METHOD_SCORE}")
COLLECT_ARGS+=(--expected-method "$METHOD")

srun "$SCIB_PYTHON" "${REPO_ROOT}/scripts/collect_eye4_scib113_scores.py" \
  "${COLLECT_ARGS[@]}" \
  --output "$COMBINED"

srun "$PYTHON" "${REPO_ROOT}/scripts/plot_eye4_scored_umaps.py" \
  --graph "PCA=${NEW_SCIB}/precomputed_graphs/PCA.h5ad" \
  --graph "Random=${NEW_SCIB}/precomputed_graphs/Random-seed42.h5ad" \
  --graph "scPRINT-2 ZS=${NEW_SCIB}/precomputed_graphs/scPRINT-2-ZS-human-target-cell-token.h5ad" \
  --graph "scPRINT-2 FT=${NEW_SCIB}/precomputed_graphs/scPRINT-2-FT-human-target-cell-token-MMD003.h5ad" \
  --graph "TranscriptFormer=${NEW_SCIB}/precomputed_graphs/TranscriptFormer-Metazoa-human-target.h5ad" \
  --graph "scPRINT-2 ZS all-except-organism=${NEW_SCIB}/precomputed_graphs/${METHOD}.h5ad" \
  --output-dir "$FIGURES" \
  --random-state 42 \
  --min-dist 0.5 \
  --spread 1.0

test -s "$CONCAT"
test -s "${CONCAT%.h5ad}.metadata.json"
test -s "$COMBINED"
test -s "${COMBINED%.csv}.metadata.json"
test -s "${COMBINED%.csv}.COMPLETE"
test -s "${FIGURES}/eye4_scored_umaps_overview.png"
test -s "${FIGURES}/eye4_scored_umaps.metadata.json"
test -s "${FIGURES}/UMAPS.COMPLETE"
printf "COMPLETE\n" > "${NEW_SCIB}/RUN.COMPLETE"
