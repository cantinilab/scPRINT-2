#!/bin/bash
#SBATCH --job-name=scprint2-mmd-score
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_dkd_ab_20260815/logs/score-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_dkd_ab_20260815/logs/score-%j.err
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread

source /etc/profile
set -eo pipefail
module load cuda/12.2.0
module load r/4.4.1

repo=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
root=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/scprint2_mmd_dkd_ab_20260815
work_root=/lustre/fswork/projects/rech/xeg/uat95fg
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OP_SOLUTION_ROOT=/lustre/fsn1/projects/rech/xeg/uat95fg/openproblems_reconstructed
export R_LIBS_USER="${R_LIBS_USER:-$work_root/R/library/4.4}"

for mode in fixed off; do
  "$repo/.venv/bin/python" "$repo/scripts/score_openproblems_embedding.py" \
    --input "$root/$mode/dkd_scprint2_small_v2_finetune_cell_type_classification_output.h5ad" \
    --output "$root/$mode/dkd_scprint2_small_v2_finetune_cell_type_op_scib.csv" \
    --dataset cellxgene_census/dkd \
    --embedding-key scprint_emb \
    --trust-positional-order \
    --method-id "scprint2_ft_mmd_$mode"
done
