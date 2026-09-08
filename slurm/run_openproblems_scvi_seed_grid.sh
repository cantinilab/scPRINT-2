#!/bin/bash
#SBATCH --job-name=scvi-seed-grid
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/openproblems_scvi_seed_grid/dkd/logs/seed-%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/openproblems_scvi_seed_grid/dkd/logs/seed-%A_%a.err
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-2

set -eo pipefail
source /etc/profile
set -u

repo=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
root=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/openproblems_scvi_seed_grid/dkd
seed="$SLURM_ARRAY_TASK_ID"
output="$root/seed_$seed"

mkdir -p "$root/logs" "$output"
export REPO_ROOT="$repo"
export SCVI_OVERLAY=/lustre/fswork/projects/rech/xeg/uat95fg/scvi-op-overlay
export SCVI_SEED="$seed"

bash "$repo/slurm/run_openproblems_scvi_validation.sh" dkd "$output"
