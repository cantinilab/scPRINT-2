#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm/task3-fresh-transcriptformer-metazoa-%j.out
#SBATCH --error=slurm/task3-fresh-transcriptformer-metazoa-%j.out

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
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
CHECKPOINT="${WORK_ROOT}/models/transcriptformer/tf_metazoa"
RUN_ID="task3_fresh_transcriptformer_metazoa_scib113_20260817"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/${RUN_ID}"
OUTPUT="${ROOT}/transcriptformer_metazoa_fresh.h5ad"

if [[ -e "$ROOT" ]]; then
  echo "Refusing to reuse an existing fresh-run root: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"

test -s "$SOURCE"
test -x "$TF_PYTHON"
test -s "$CHECKPOINT/config.json"
test -s "$CHECKPOINT/model_weights.pt"
test -s "$CHECKPOINT/vocabs/mus_musculus_gene.h5"

export PYTHONPATH="${SCRATCH_ROOT}/scprint_data/setuptools-overlay${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

srun "$TF_PYTHON" "$REPO_ROOT/scripts/run_task3_cross_species_embedding.py" transcriptformer \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-key orig.ident \
  --cell-type-key cell_type_ontology_term_id \
  --batch-size 32 \
  --num-workers 8 \
  --feature-organism NCBITaxon:10090 \
  --notebook-provenance "fresh Task3 TranscriptFormer Metazoa inference for the scib 1.1.3 precomputed-KNN rerun"

test -s "$OUTPUT"
test -s "${OUTPUT%.h5ad}.metadata.json"

SOURCE_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
CONFIG_SHA256="$(sha256sum "$CHECKPOINT/config.json" | awk '{print $1}')"
WEIGHTS_SHA256="$(sha256sum "$CHECKPOINT/model_weights.pt" | awk '{print $1}')"
MOUSE_VOCAB_SHA256="$(sha256sum "$CHECKPOINT/vocabs/mus_musculus_gene.h5" | awk '{print $1}')"
OUTPUT_SHA256="$(sha256sum "$OUTPUT" | awk '{print $1}')"

cat >"$ROOT/fresh_embedding_manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "source": {"path": "$SOURCE", "sha256": "$SOURCE_SHA256"},
  "checkpoint": {
    "path": "$CHECKPOINT",
    "config_sha256": "$CONFIG_SHA256",
    "model_weights_sha256": "$WEIGHTS_SHA256",
    "mouse_vocab_sha256": "$MOUSE_VOCAB_SHA256"
  },
  "embedding": {"path": "$OUTPUT", "key": "model_emb", "sha256": "$OUTPUT_SHA256"}
}
EOF

touch "$ROOT/EMBEDDING_COMPLETE"
