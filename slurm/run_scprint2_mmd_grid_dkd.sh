#!/bin/bash
#SBATCH --job-name=scprint2-mmd-grid
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_grid_dkd_20260815/logs/grid-%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_grid_dkd_20260815/logs/grid-%A_%a.err
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3

set -eo pipefail
source /etc/profile
set -u
module load cuda/12.2.0

repo=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
root=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_grid_dkd_20260815
weights=(0.003 0.01 0.03 0.1)
tags=(0p003 0p01 0p03 0p1)
weight="${weights[$SLURM_ARRAY_TASK_ID]}"
tag="${tags[$SLURM_ARRAY_TASK_ID]}"
output="$root/mmd_$tag"

mkdir -p "$root/logs" "$output"
export MMD_MODE=fixed
export MMD_WEIGHT="$weight"
export REPO_ROOT="$repo"

bash "$repo/slurm/run_scprint2_op_full_no_assay.sh" dkd finetune "$output"
