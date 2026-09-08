#!/bin/bash
#SBATCH --job-name=op-scib-existing
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/logs/op-scib-existing-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/logs/op-scib-existing-%j.err
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread

set -eo pipefail
source /etc/profile
set -u
module load r/4.4.1

: "${INPUT_PATH:?missing INPUT_PATH}"
: "${OUTPUT_PATH:?missing OUTPUT_PATH}"
: "${DATASET:?missing DATASET}"
: "${METHOD_ID:?missing METHOD_ID}"

repo=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
scratch=/lustre/fsn1/projects/rech/xeg/uat95fg
embedding_key="${EMBEDDING_KEY:-scprint_emb}"

test -s "$INPUT_PATH"
mkdir -p "$(dirname "$OUTPUT_PATH")" "$scratch/scprint_data/logs"
export PYTHONPATH="$scratch/scprint_data/setuptools-overlay:$repo${PYTHONPATH:+:$PYTHONPATH}"
export R_HOME="$(R RHOME)"
export R_LIBS_USER=/lustre/fswork/projects/rech/xeg/uat95fg/R/library/4.4
export SCIB_NATIVE_CACHE=/lustre/fswork/projects/rech/xeg/uat95fg/.cache/scib-native
export OP_SOLUTION_ROOT="$scratch/openproblems_reconstructed"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$repo"
extra_args=()
if [[ "${TRUST_POSITIONAL_ORDER:-0}" == "1" ]]; then
  extra_args+=(--trust-positional-order)
fi
"$repo/.venv/bin/python" scripts/score_openproblems_embedding.py \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --dataset "cellxgene_census/$DATASET" \
  --embedding-key "$embedding_key" \
  --method-id "$METHOD_ID" \
  "${extra_args[@]}"
