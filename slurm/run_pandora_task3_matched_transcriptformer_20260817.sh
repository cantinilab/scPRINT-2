#!/bin/bash
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm/pandora-task3-matched-transcriptformer-%j.out
#SBATCH --error=slurm/pandora-task3-matched-transcriptformer-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0
module unload cuda/12.2.0 || true

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
TF_PYTHON="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}/bin/python"
SOURCE="${PANDORA_INPUT:-${WORK_ROOT}/data/pandora_lung/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad}"
CHECKPOINT="${WORK_ROOT}/models/transcriptformer/tf_metazoa"
RUN_ID="pandora_task3_matched_20260817"
ROOT="${WORK_ROOT}/data/pandora_lung/${RUN_ID}"
OUTPUT="${ROOT}/transcriptformer_metazoa_fresh.h5ad"

test -s "$SOURCE"
test -x "$TF_PYTHON"
test -s "$CHECKPOINT/config.json"
test -s "$CHECKPOINT/model_weights.pt"
mkdir -p "$ROOT"
if [[ -e "$OUTPUT" || -e "$ROOT/TRANSCRIPTFORMER_EMBEDDING_COMPLETE" ]]; then
  echo "Refusing to reuse or overwrite fresh TranscriptFormer output" >&2
  exit 3
fi

export PYTHONPATH="${SCRATCH_ROOT}/scprint_data/setuptools-overlay${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$REPO_ROOT"

srun "$TF_PYTHON" scripts/run_task3_cross_species_embedding.py transcriptformer \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-key species \
  --cell-type-key cell_type_ontology_term_id \
  --batch-size 32 \
  --num-workers 8 \
  --feature-organism NCBITaxon:10090 \
  --seed 42 \
  --notebook-provenance \
  "Pandora fresh TranscriptFormer Metazoa inference matched to the task3 runner"

test -s "$OUTPUT"
test -s "${OUTPUT%.h5ad}.metadata.json"
touch "$ROOT/TRANSCRIPTFORMER_EMBEDDING_COMPLETE"
