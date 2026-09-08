#!/bin/bash
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=slurm/pandora-task3-matched-umaps-%j.out
#SBATCH --error=slurm/pandora-task3-matched-umaps-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PLOT_PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
MAP_PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RUN_ROOT="${WORK_ROOT}/data/pandora_lung/pandora_task3_matched_20260817"
GRAPH_ROOT="${RUN_ROOT}/scores/precomputed_graphs"
OUTPUT_DIR="${RUN_ROOT}/umaps_20260818"
ONTOLOGY_PARQUET="${REPO_ROOT}/data/results/cross_species_embedding/task3_fresh_umaps_20260818/df_all__cl__2025-12-17__CellType.parquet"
ONTOLOGY_JSON="${RUN_ROOT}/cell_ontology_name_map_2025-12-17.json"

test ! -e "${OUTPUT_DIR}"
test -x "${PLOT_PYTHON}"
test -x "${MAP_PYTHON}"
test -s "${ONTOLOGY_PARQUET}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"${MAP_PYTHON}" -c 'import json, pandas as pd, sys; source, output = sys.argv[1:]; frame = pd.read_parquet(source); frame = frame.set_index("ontology_id") if frame.index.name != "ontology_id" else frame; mapping = frame["name"].dropna().astype(str).to_dict(); open(output, "w", encoding="utf-8").write(json.dumps(mapping, indent=2, sort_keys=True) + "\n")' "${ONTOLOGY_PARQUET}" "${ONTOLOGY_JSON}"

srun "${PLOT_PYTHON}" "${REPO_ROOT}/scripts/plot_pandora_task3_matched_umaps.py" \
  --pca-graph "${GRAPH_ROOT}/PCA-expression-CP10K-log1p-task3-matched.h5ad" \
  --scprint2-graph "${GRAPH_ROOT}/scPRINT-2-ZS-cell-token-task3-matched.h5ad" \
  --scprint2-ft-graph "${GRAPH_ROOT}/scPRINT-2-FT-cell-token-mmd003-task3-matched.h5ad" \
  --transcriptformer-graph "${GRAPH_ROOT}/TranscriptFormer-Metazoa-task3-matched.h5ad" \
  --scprint2-predictions "${RUN_ROOT}/scprint2_zero_shot_small_v2_knn.h5ad" \
  --scprint2-ft-predictions "${RUN_ROOT}/scprint2_ft_small_v2_mmd003_knn_with_organism_classification.h5ad" \
  --ontology-map-json "${ONTOLOGY_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --random-state 42 \
  --min-dist 0.5 \
  --spread 1.0 \
  --prediction-top-n 24

test -e "${OUTPUT_DIR}/COMPLETE"
