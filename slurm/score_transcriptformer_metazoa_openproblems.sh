#!/bin/bash
#SBATCH --job-name=tf-metazoa-scib
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/transcriptformer_metazoa_op_20260815/logs/score-%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/transcriptformer_metazoa_op_20260815/logs/score-%A_%a.err
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3

set -eo pipefail
source /etc/profile
set -u
module load r/4.4.1

repo=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
scratch=/lustre/fsn1/projects/rech/xeg/uat95fg
root="$scratch/scprint_data/transcriptformer_metazoa_op_20260815"
datasets=(dkd gtex_v9 hypomap mouse_pancreas_atlas)
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
embedding="$root/$dataset/${dataset}_tf_metazoa_embeddings.h5ad"
scores="$root/$dataset/${dataset}_tf_metazoa_op_scib.csv"

test -f "$embedding"
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
"$repo/.venv/bin/python" scripts/score_openproblems_embedding.py \
  --input "$embedding" \
  --output "$scores" \
  --dataset "cellxgene_census/$dataset" \
  --embedding-key model_emb \
  --method-id transcriptformer_metazoa
