#!/bin/bash
#SBATCH --job-name=eye4-scprint2
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_matched_20260823/logs/scprint2-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_matched_20260823/logs/scprint2-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_matched_20260823}"
SOURCE="${ROOT}/eye4_mouse_one2one_scprint_preprocessed.h5ad"
CHECKPOINT="${SMALL_V2_CHECKPOINT:-${WORK_ROOT}/models/small-v2.ckpt}"
ZERO_SHOT="${ROOT}/embeddings/scprint2_zero_shot_small_v2_knn.h5ad"
FINE_TUNED="${ROOT}/embeddings/scprint2_ft_small_v2_mmd003_knn.h5ad"
BASELINES="${ROOT}/embeddings/expression_pca50_random_seed42.h5ad"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=offline
export WANDB_DISABLED=true

test -s "$SOURCE"
test -s "$CHECKPOINT"

if [[ ! -s "$BASELINES" || ! -s "${BASELINES%.h5ad}.metadata.json" ]]; then
  test ! -e "$BASELINES"
  test ! -e "${BASELINES%.h5ad}.metadata.json"
  srun "$PYTHON" "${REPO_ROOT}/scripts/build_expression_scib_baselines.py" \
    --input "$SOURCE" \
    --output "$BASELINES" \
    --seed 42
fi

if [[ ! -s "$ZERO_SHOT" || ! -s "${ZERO_SHOT%.h5ad}.metadata.json" ]]; then
  test ! -e "$ZERO_SHOT"
  test ! -e "${ZERO_SHOT%.h5ad}.metadata.json"
  srun "$PYTHON" "${REPO_ROOT}/scripts/run_task3_cross_species_embedding.py" \
    scprint2-zero-shot \
    --input "$SOURCE" \
    --checkpoint "$CHECKPOINT" \
    --output "$ZERO_SHOT" \
    --batch-key species \
    --cell-type-key cell_type_ontology_term_id \
    --embed-how "random expr" \
    --embed-max-len 2800 \
    --use-knn \
    --rebuild-knn-graph \
    --seed 42 \
    --notebook-provenance \
    "eye4 GEO raw counts; strict mouse one-to-one genes; small-v2 zero-shot cell-type token; multi-cell kNN"
fi

if [[ ! -s "$FINE_TUNED" || ! -s "${FINE_TUNED%.h5ad}.metadata.json" ]]; then
  test ! -e "$FINE_TUNED"
  test ! -e "${FINE_TUNED%.h5ad}.metadata.json"
  srun "$PYTHON" "${REPO_ROOT}/scripts/run_task3_cross_species_embedding.py" \
    scprint2-ft \
    --input "$SOURCE" \
    --checkpoint "$CHECKPOINT" \
    --output "$FINE_TUNED" \
    --batch-key species \
    --cell-type-key cell_type_ontology_term_id \
    --finetune-max-len 2200 \
    --embed-max-len 2800 \
    --num-epochs 8 \
    --mmd-target species \
    --mmd-scale 0.03 \
    --use-knn \
    --rebuild-knn-graph \
    --include-unknown-train-labels \
    --seed 42 \
    --notebook-provenance \
    "eye4 GEO raw counts; strict mouse one-to-one genes; differentiable balanced four-species MMD=0.03 on cell-type token; best validation checkpoint restored; multi-cell kNN"
fi

test -s "$BASELINES"
test -s "$ZERO_SHOT"
test -s "${ZERO_SHOT%.h5ad}.metadata.json"
test -s "$FINE_TUNED"
test -s "${FINE_TUNED%.h5ad}.metadata.json"
sha256sum "$BASELINES" "$ZERO_SHOT" "$FINE_TUNED" > "${ROOT}/embeddings/scprint2_sha256.tsv"
touch "${ROOT}/embeddings/SCPRINT2.COMPLETE"
