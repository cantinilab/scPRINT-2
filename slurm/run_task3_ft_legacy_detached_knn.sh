#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-ft-legacy-detached-knn-%j.out
#SBATCH --error=slurm/task3-ft-legacy-detached-knn-%j.out

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
OUTPUT="${ROOT}/scprint2_ft_legacy_detached_knn.h5ad"
LEGACY_SCORES="${ROOT}/task3_scib051_ft_legacy_detached_knn.csv"
MODERN_SCORES="${ROOT}/task3_modern_ft_legacy_detached_knn.csv"

test -s "$CHECKPOINT"
test -s "$SCIB_OVERLAY/scib_metrics/__init__.py"

srun "$PYTHON" "$REPO_ROOT/scripts/run_task3_cross_species_embedding.py" \
  scprint-mmd \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --finetune-max-len 2200 \
  --embed-max-len 2800 \
  --num-epochs 8 \
  --use-knn \
  --legacy-detached-mmd \
  --no-restore-best-model \
  --include-unknown-train-labels \
  --seed 42 \
  --notebook-provenance \
  "fine_tuning_cross_species_emb_mmd.ipynb legacy detached-MMD; train max_len=2200; inference max_len=2800; multi-cell kNN"

# The historical notebook used scib-metrics 0.5.1 and a CPU JAX backend.
JAX_PLATFORMS=cpu PYTHONPATH="${SCIB_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$LEGACY_SCORES" \
  --batch-key batch \
  --label-key cell_type_ontology_term_id \
  --seed 42 \
  --embedding "scPRINT-2-FT-legacy-detached-knn=${OUTPUT}:scprint_emb" \
  --embedding "scPRINT-2-FT-cell-token-legacy-detached-knn=${OUTPUT}:scprint_emb_cell_type_ontology_term_id"

JAX_PLATFORMS=cpu "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$MODERN_SCORES" \
  --seed 42 \
  --embedding "scPRINT-2-FT-legacy-detached-knn=${OUTPUT}:scprint_emb" \
  --embedding "scPRINT-2-FT-cell-token-legacy-detached-knn=${OUTPUT}:scprint_emb_cell_type_ontology_term_id"
