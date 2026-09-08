#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=slurm/task3-fresh-transcriptformer-scib113-%j.out
#SBATCH --error=slurm/task3-fresh-transcriptformer-scib113-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
export R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
RUN_ID="task3_fresh_transcriptformer_metazoa_scib113_20260817"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/${RUN_ID}"
EMBEDDING="${ROOT}/transcriptformer_metazoa_fresh.h5ad"
OUTPUT="${ROOT}/task3_transcriptformer_scib113_precomputed_scores.csv"

test -f "$ROOT/EMBEDDING_COMPLETE"
test -s "$EMBEDDING"

srun "$PYTHON" "$REPO_ROOT/scripts/benchmark_task3_scib113_precomputed.py" \
  --source "$SOURCE" \
  --batch-key orig.ident \
  --label-key cell_type_ontology_term_id \
  --n-cores "${SLURM_CPUS_PER_TASK:-32}" \
  --embedding "TranscriptFormer-Metazoa-fresh-scib113=${EMBEDDING}:model_emb" \
  --output "$OUTPUT"
