#!/bin/bash
#SBATCH --job-name=eye4-human-tf
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/baseline-tf-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/baseline-tf-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
SCPRINT_PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
TF_PYTHON="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}/bin/python"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_human_target_hybrid_20260824}"
SOURCE="${ROOT}/eye4_human_ensembl_small_v2.h5ad"
CHECKPOINT="${WORK_ROOT}/models/transcriptformer/tf_metazoa"
BASELINES="${ROOT}/embeddings/expression_pca50_random_seed42.h5ad"
OUTPUT="${ROOT}/embeddings/transcriptformer_metazoa_human_target.h5ad"
export PYTHONPATH="${REPO_ROOT}:${SCRATCH_ROOT}/scprint_data/setuptools-overlay${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${ROOT}/embeddings"

test -s "$SOURCE"
test -s "${ROOT}/HUMAN_TARGET_PREPARE.COMPLETE"
test -x "$SCPRINT_PYTHON"
test -x "$TF_PYTHON"
test -s "$CHECKPOINT/config.json"
test -s "$CHECKPOINT/model_weights.pt"
test -s "$CHECKPOINT/vocabs/homo_sapiens_gene.h5"

if [[ ! -s "$BASELINES" || ! -s "${BASELINES%.h5ad}.metadata.json" ]]; then
  test ! -e "$BASELINES"
  test ! -e "${BASELINES%.h5ad}.metadata.json"
  srun "$SCPRINT_PYTHON" "${REPO_ROOT}/scripts/build_expression_scib_baselines.py" \
    --input "$SOURCE" \
    --output "$BASELINES" \
    --seed 42
fi

if [[ ! -s "$OUTPUT" || ! -s "${OUTPUT%.h5ad}.metadata.json" ]]; then
  test ! -e "$OUTPUT"
  test ! -e "${OUTPUT%.h5ad}.metadata.json"
  srun "$TF_PYTHON" "${REPO_ROOT}/scripts/run_task3_cross_species_embedding.py" \
    transcriptformer \
    --input "$SOURCE" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --batch-key species \
    --cell-type-key cell_type_ontology_term_id \
    --batch-size 32 \
    --num-workers 8 \
    --feature-organism NCBITaxon:9606 \
    --notebook-provenance \
    "eye4 human-target union: Human/Macaque/Pig HGNC-like symbols mapped directly to human ENSG, Mouse mapped by high-confidence orthology; no cross-species gene intersection"
fi

test -s "$BASELINES"
test -s "${BASELINES%.h5ad}.metadata.json"
test -s "$OUTPUT"
test -s "${OUTPUT%.h5ad}.metadata.json"
sha256sum "$SOURCE" "$BASELINES" "$CHECKPOINT/config.json" \
  "$CHECKPOINT/model_weights.pt" "$CHECKPOINT/vocabs/homo_sapiens_gene.h5" \
  "$OUTPUT" > "${ROOT}/embeddings/baseline_transcriptformer_human_sha256.tsv"
touch "${ROOT}/embeddings/BASELINES_TRANSCRIPTFORMER_HUMAN.COMPLETE"
