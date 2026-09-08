#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-historical-noop-ft-embed2800-knn-%j.out
#SBATCH --error=slurm/task3-historical-noop-ft-embed2800-knn-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SCIB_OVERLAY="${SCIB_051_OVERLAY:-${WORK_ROOT}/scib_metrics_0_5_1_overlay}"
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3"
CHECKPOINT="${WORK_ROOT}/models/small-v2.ckpt"
OUTPUT="${ROOT}/scprint2_historical_noop_ft_embed2800_knn.h5ad"
MODERN_SCORES="${ROOT}/task3_modern_historical_noop_ft_embed2800_knn.csv"
LEGACY_SCORES="${ROOT}/task3_scib051_historical_noop_ft_embed2800_knn.csv"

test -s "$CHECKPOINT"
test -s "$SCIB_OVERLAY/scib_metrics/__init__.py"

# The historical notebook's optimization loop is fully commented out. Its saved
# fit_4.ckpt therefore retains the small-v2 model weights; only the inference
# configuration below (max_len=2800 and metacell kNN inputs) affects embeddings.
srun "$PYTHON" "$REPO_ROOT/scripts/run_task3_cross_species_embedding.py" \
  scprint-zero-shot \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --embed-how "random expr" \
  --embed-max-len 2800 \
  --use-knn \
  --seed 42 \
  --notebook-provenance \
  "fine_tuning_cross_species_emb_mmd.ipynb historical no-op training; small-v2; inference max_len=2800; multi-cell kNN"

"$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$MODERN_SCORES" \
  --seed 42 \
  --embedding "scPRINT-2-historical-noop-ft=${OUTPUT}:scprint_emb" \
  --embedding "scPRINT-2-historical-noop-ft-cell-token=${OUTPUT}:scprint_emb_cell_type_ontology_term_id"

PYTHONPATH="${SCIB_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$LEGACY_SCORES" \
  --batch-key batch \
  --label-key cell_type_ontology_term_id \
  --seed 42 \
  --embedding "scPRINT-2-historical-noop-ft=${OUTPUT}:scprint_emb" \
  --embedding "scPRINT-2-historical-noop-ft-cell-token=${OUTPUT}:scprint_emb_cell_type_ontology_term_id"
