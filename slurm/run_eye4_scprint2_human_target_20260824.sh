#!/bin/bash
#SBATCH --job-name=eye4-human-scprint2
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/scprint2-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/scprint2-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_human_target_hybrid_20260824}"
SOURCE="${ROOT}/eye4_human_ensembl_small_v2.h5ad"
CHECKPOINT="${SMALL_V2_CHECKPOINT:-${WORK_ROOT}/models/small-v2.ckpt}"
ZERO_SHOT="${ROOT}/embeddings/scprint2_zero_shot_small_v2_human_target_knn.h5ad"
FINE_TUNED="${ROOT}/embeddings/scprint2_ft_small_v2_human_target_mmd003_knn.h5ad"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=offline
export WANDB_DISABLED=true
mkdir -p "${ROOT}/embeddings"

test -s "$SOURCE"
test -s "$CHECKPOINT"
test -s "${ROOT}/HUMAN_TARGET_PREPARE.COMPLETE"

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
    --align-checkpoint-genes \
    --seed 42 \
    --notebook-provenance \
    "eye4 human-target ENSG union; all cells declared human feature space; small-v2 zero-shot cell-type token; multi-cell kNN"
fi

if [[ ! -s "$FINE_TUNED" || ! -s "${FINE_TUNED%.h5ad}.metadata.json" ]]; then
  test ! -e "$FINE_TUNED"
  test ! -e "${FINE_TUNED%.h5ad}.metadata.json"
  test ! -e "${FINE_TUNED%.h5ad}.ckpt"
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
    --align-checkpoint-genes \
    --require-known-train-labels \
    --seed 42 \
    --notebook-provenance \
    "eye4 human-target ENSG union; all 13 labels known by small-v2; differentiable balanced four-species MMD=0.03 on cell-type token; best checkpoint restored; multi-cell kNN"
fi

test -s "$ZERO_SHOT"
test -s "${ZERO_SHOT%.h5ad}.metadata.json"
test -s "$FINE_TUNED"
test -s "${FINE_TUNED%.h5ad}.metadata.json"
test -s "${FINE_TUNED%.h5ad}.ckpt"
sha256sum "$SOURCE" "$ZERO_SHOT" "$FINE_TUNED" \
  > "${ROOT}/embeddings/scprint2_human_target_sha256.tsv"
touch "${ROOT}/embeddings/SCPRINT2_HUMAN_TARGET.COMPLETE"
