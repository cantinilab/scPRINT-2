#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=slurm/task3-ft-detached-embed2800-%j.out
#SBATCH --error=slurm/task3-ft-detached-embed2800-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3"
CHECKPOINT="${ROOT}/scprint_mmd_embeddings.ckpt"
OUTPUT="${ROOT}/scprint_mmd_detached_embed2800.h5ad"
SCORES="${ROOT}/task3_modern_ft_detached_embed2800.csv"

test -s "$CHECKPOINT"

srun "$PYTHON" "$REPO_ROOT/scripts/run_task3_cross_species_embedding.py" \
  scprint-zero-shot \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --embed-how "random expr" \
  --embed-max-len 2800 \
  --seed 42 \
  --notebook-provenance \
  "fine_tuning_cross_species_emb_mmd.ipynb detached-MMD checkpoint; inference max_len=2800"

"$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_modern_scib.py" \
  --source "$SOURCE" \
  --output "$SCORES" \
  --seed 42 \
  --embedding "scPRINT-2-FT-detached-maxlen2800=${OUTPUT}:scprint_emb" \
  --embedding "scPRINT-2-FT-cell-token-detached-maxlen2800=${OUTPUT}:scprint_emb_cell_type_ontology_term_id"
