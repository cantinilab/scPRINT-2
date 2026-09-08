#!/bin/bash
#SBATCH --job-name=tf-metazoa-op
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/transcriptformer_metazoa_op_20260815/logs/run-%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/transcriptformer_metazoa_op_20260815/logs/run-%A_%a.err
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3

set -eo pipefail
source /etc/profile
set -u
module load r/4.4.1

repo=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
scratch=/lustre/fsn1/projects/rech/xeg/uat95fg
root="$scratch/scprint_data/transcriptformer_metazoa_op_20260815"
tf_python="$scratch/venvs/transcriptformer-h100-0.6.1/bin/python"
checkpoint=/lustre/fswork/projects/rech/xeg/uat95fg/models/transcriptformer/tf_metazoa
datasets=(dkd gtex_v9 hypomap mouse_pancreas_atlas)
organisms=(NCBITaxon:9606 NCBITaxon:9606 NCBITaxon:10090 NCBITaxon:10090)
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
organism="${organisms[$SLURM_ARRAY_TASK_ID]}"
output_dir="$root/$dataset"
embedding="$output_dir/${dataset}_tf_metazoa_embeddings.h5ad"
scores="$output_dir/${dataset}_tf_metazoa_op_scib.csv"

mkdir -p "$root/logs" "$output_dir"
export PYTHONPATH="$scratch/scprint_data/setuptools-overlay:$repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
if [[ ! -f "$embedding" ]]; then
  "$tf_python" scripts/run_task3_cross_species_embedding.py transcriptformer \
    --input "$repo/data/temp/cellxgene_census/${dataset}_proc.h5ad" \
    --checkpoint "$checkpoint" \
    --output "$embedding" \
    --feature-organism "$organism" \
    --batch-size 32 \
    --num-workers 2 \
    --notebook-provenance \
      notebooks/scPRINT-2-repro-notebooks/cross-species-embbedding-transcriptformer-metazoa.ipynb
fi

if [[ ! -f "$scores" ]]; then
  "$repo/.venv/bin/python" scripts/score_openproblems_embedding.py \
    --input "$embedding" \
    --output "$scores" \
    --dataset "cellxgene_census/$dataset" \
    --embedding-key model_emb \
    --method-id transcriptformer_metazoa
fi
