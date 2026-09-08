#!/bin/bash
#SBATCH --job-name=eye4-tf
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_matched_20260823/logs/transcriptformer-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_matched_20260823/logs/transcriptformer-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
TF_PYTHON="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}/bin/python"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_matched_20260823}"
SOURCE="${ROOT}/eye4_mouse_one2one_scprint_preprocessed.h5ad"
CHECKPOINT="${WORK_ROOT}/models/transcriptformer/tf_metazoa"
OUTPUT="${ROOT}/embeddings/transcriptformer_metazoa.h5ad"
export PYTHONPATH="${REPO_ROOT}:${SCRATCH_ROOT}/scprint_data/setuptools-overlay${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test -s "$SOURCE"
test -x "$TF_PYTHON"
test -s "$CHECKPOINT/config.json"
test -s "$CHECKPOINT/model_weights.pt"
test -s "$CHECKPOINT/vocabs/mus_musculus_gene.h5"
test ! -e "$OUTPUT"

srun "$TF_PYTHON" "${REPO_ROOT}/scripts/run_task3_cross_species_embedding.py" \
  transcriptformer \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-key species \
  --cell-type-key cell_type_ontology_term_id \
  --batch-size 32 \
  --num-workers 8 \
  --feature-organism NCBITaxon:10090 \
  --notebook-provenance \
  "eye4 TranscriptFormer Metazoa inference on verified GEO raw counts mapped to strict mouse one-to-one genes"

test -s "$OUTPUT"
test -s "${OUTPUT%.h5ad}.metadata.json"
sha256sum "$SOURCE" "$CHECKPOINT/config.json" "$CHECKPOINT/model_weights.pt" \
  "$CHECKPOINT/vocabs/mus_musculus_gene.h5" "$OUTPUT" \
  > "${ROOT}/embeddings/transcriptformer_sha256.tsv"
touch "${ROOT}/embeddings/TRANSCRIPTFORMER.COMPLETE"
