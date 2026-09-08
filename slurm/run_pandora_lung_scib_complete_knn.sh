#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-scib-complete-knn-%j.out
#SBATCH --error=slurm/pandora-scib-complete-knn-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
FT="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_ft_cell_type_only_knn.h5ad"
FT_ALL="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_ft_cell_type_only_knn_token_concat_pca50.h5ad"
ZS="${DATA_ROOT}/pandora_lung_mouse_space_filtered_93423_scprint2_small_v2_knn.h5ad"
ZS_ALL="${DATA_ROOT}/pandora_lung_filtered_93423_small_v2_knn_token_concat_pca50.h5ad"
TF="${DATA_ROOT}/pandora_lung_transcriptformer.h5ad"
OUTPUT="${DATA_ROOT}/pandora_lung_filtered_93423_scib_complete_knn.tsv"

cd "$REPO_ROOT"
srun .venv/bin/python scripts/build_task3_token_concat_pca.py \
  --input "$FT" \
  --output "$FT_ALL" \
  --method scprint2_small_v2_ft_cell_type_only_knn_all_tokens_pca50
srun .venv/bin/python scripts/build_task3_token_concat_pca.py \
  --input "$ZS" \
  --output "$ZS_ALL" \
  --method scprint2_small_v2_knn_all_tokens_pca50

srun .venv/bin/python scripts/benchmark_pandora_lung_scib.py \
  --expression "${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad" \
  --embedding "$FT" \
  --embedding-key scprint_emb_cell_type_ontology_term_id \
  --embedding-name "FT cell type" \
  --additional-embedding "FT all=${FT_ALL}:token_concat_pca50" \
  --additional-embedding "ZS cell type=${ZS}:scprint_emb_cell_type_ontology_term_id" \
  --additional-embedding "ZS all=${ZS_ALL}:token_concat_pca50" \
  --additional-embedding "TranscriptFormer=${TF}:model_emb" \
  --include-random \
  --output "$OUTPUT" \
  --batch-key species \
  --label-key cell_type_ontology_term_id \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
