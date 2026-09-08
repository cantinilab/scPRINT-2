#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-scprint2-zero-shot-knn-%j.out
#SBATCH --error=slurm/task3-scprint2-zero-shot-knn-%j.out

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
CHECKPOINT="${WORK_ROOT}/models/ji9krimq.ckpt"
OUTPUT="${ROOT}/scprint2_zero_shot_knn_maxlen2800.h5ad"
SCORE_ROOT="${ROOT}/task3_scprint2_zero_shot_knn_maxlen2800"
MODERN_SCORES="${SCORE_ROOT}/task3_modern_scprint2_zero_shot_knn_maxlen2800.csv"
LEGACY_SCORES="${SCORE_ROOT}/task3_scib051_scprint2_zero_shot_knn_maxlen2800.csv"

test -s "$CHECKPOINT"
test -s "$SCIB_OVERLAY/scib_metrics/__init__.py"
mkdir -p "$SCORE_ROOT"

if [[ ! -s "$OUTPUT" ]]; then
  srun "$PYTHON" "$REPO_ROOT/scripts/run_task3_cross_species_embedding.py" \
    scprint2-zero-shot \
    --input "$SOURCE" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --embed-max-len 2800 \
    --use-knn \
    --seed 42 \
    --notebook-provenance \
    "cross-species-embbedding.ipynb; zero-shot; inference max_len=2800; multi-cell kNN"
fi

JAX_PLATFORMS=cpu "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$MODERN_SCORES" \
  --batch-key orig.ident \
  --label-key cell_type_ontology_term_id \
  --seed 42 \
  --embedding "scPRINT-2-knn=${OUTPUT}:scprint_emb" \
  --embedding "scPRINT-2-cell-token-knn=${OUTPUT}:scprint_emb_cell_type_ontology_term_id"

JAX_PLATFORMS=cpu PYTHONPATH="${SCIB_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$LEGACY_SCORES" \
  --batch-key orig.ident \
  --label-key cell_type_ontology_term_id \
  --seed 42 \
  --embedding "scPRINT-2-knn=${OUTPUT}:scprint_emb" \
  --embedding "scPRINT-2-cell-token-knn=${OUTPUT}:scprint_emb_cell_type_ontology_term_id"

touch "$SCORE_ROOT/COMPLETE"
