#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/task3-cross-species-%j.out
#SBATCH --error=slurm/task3-cross-species-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

REPO_ROOT="${SCPRINT_REPO:-/lustre/fswork/projects/rech/xeg/${USER}/scPRINT}"
MODE="$1"
OUTPUT="$2"
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
INPUT="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"

case "$MODE" in
  scprint-zero-shot|scprint2-zero-shot)
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    CHECKPOINT="${WORK_ROOT}/models/ji9krimq.ckpt"
    NOTEBOOK="notebooks/scPRINT-2-repro-notebooks/cross-species-embbedding.ipynb"
    ;;
  scprint1-zero-shot)
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    CHECKPOINT="${WORK_ROOT}/models/ogvvg2z7-v1.ckpt"
    NOTEBOOK="paper-task3 runner using the corrected scPRINT-1 checkpoint vocabulary"
    ;;
  scprint-mmd|scprint2-ft)
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    CHECKPOINT="${WORK_ROOT}/models/small-v2.ckpt"
    NOTEBOOK="notebooks/scPRINT-2-repro-notebooks/fine_tuning_cross_species_emb_mmd.ipynb"
    ;;
  transcriptformer)
    module unload cuda/12.2.0 || true
    SCRATCH_ROOT="${SCRATCH:-/lustre/fsn1/projects/rech/xeg/${USER}}"
    PYTHON="${TF_ENV:-${SCRATCH_ROOT}/venvs/transcriptformer-h100-0.6.1}/bin/python"
    CHECKPOINT="${WORK_ROOT}/models/transcriptformer/tf_metazoa"
    NOTEBOOK="notebooks/scPRINT-2-repro-notebooks/cross-species-embbedding-transcriptformer-metazoa.ipynb"
    export PYTHONPATH="${SCRATCH_ROOT}/scprint_data/setuptools-overlay${PYTHONPATH:+:${PYTHONPATH}}"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"
EXTRA_ARGS=()
if [[ "$MODE" == "transcriptformer" ]]; then
  EXTRA_ARGS+=(--feature-organism NCBITaxon:10090)
fi
srun "$PYTHON" scripts/run_task3_cross_species_embedding.py "$MODE" \
  --input "$INPUT" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --notebook-provenance "$NOTEBOOK" \
  "${EXTRA_ARGS[@]}"
