#!/bin/bash
#SBATCH --job-name=scprint2-mmd003-op
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_0p03_op_20260815/logs/run-%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_0p03_op_20260815/logs/run-%A_%a.err
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-2

set -eo pipefail
source /etc/profile
set -u
module load cuda/12.2.0

repo=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
root=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_0p03_op_20260815
datasets=(gtex_v9 hypomap mouse_pancreas_atlas)
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
output="$root/$dataset"

mkdir -p "$root/logs" "$output"
export MMD_MODE=fixed
export MMD_WEIGHT=0.03
export REPO_ROOT="$repo"

bash "$repo/slurm/run_scprint2_op_full_no_assay.sh" "$dataset" finetune "$output"
