#!/bin/bash
#SBATCH --partition=gpu_p6
#SBATCH --account=xeg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=slurm/pandora-task3-matched-scprint2-%j.out
#SBATCH --error=slurm/pandora-task3-matched-scprint2-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SOURCE="${PANDORA_INPUT:-${WORK_ROOT}/data/pandora_lung/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad}"
CHECKPOINT="${SMALL_V2_CHECKPOINT:-${WORK_ROOT}/models/small-v2.ckpt}"
RUN_ID="pandora_task3_matched_20260817"
ROOT="${WORK_ROOT}/data/pandora_lung/${RUN_ID}"
ZERO_SHOT="${ROOT}/scprint2_zero_shot_small_v2_knn.h5ad"
FINE_TUNED="${ROOT}/scprint2_ft_small_v2_mmd003_knn_with_organism_classification.h5ad"
ZERO_SHOT_CONCAT="${ROOT}/scprint2_zero_shot_all_tokens_except_assay_organism_pca50.h5ad"
FINE_TUNED_CONCAT="${ROOT}/scprint2_ft_all_tokens_except_assay_organism_pca50.h5ad"
BASELINES="${ROOT}/expression_pca_and_random.h5ad"

test -s "$SOURCE"
test -s "$CHECKPOINT"
mkdir -p "$ROOT"
if [[ -e "$ZERO_SHOT" || -e "$FINE_TUNED" || -e "$ZERO_SHOT_CONCAT" || \
      -e "$FINE_TUNED_CONCAT" || -e "$BASELINES" || \
      -e "$ROOT/SCPRINT2_EMBEDDINGS_COMPLETE" ]]; then
  echo "Refusing to reuse or overwrite fresh scPRINT-2 outputs in: $ROOT" >&2
  exit 3
fi
cd "$REPO_ROOT"

srun "$PYTHON" scripts/run_task3_cross_species_embedding.py \
  scprint2-zero-shot \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$ZERO_SHOT" \
  --batch-key species \
  --cell-type-key cell_type_ontology_term_id \
  --embed-how "random expr" \
  --embed-max-len 2800 \
  --align-checkpoint-genes \
  --use-knn \
  --rebuild-knn-graph \
  --seed 42 \
  --notebook-provenance \
  "Pandora fresh rerun matched to task3: small-v2 zero-shot; ontology labels; forced CP10K/log1p/PCA/neighbors; multi-cell kNN"

srun "$PYTHON" scripts/run_task3_cross_species_embedding.py \
  scprint2-ft \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$FINE_TUNED" \
  --batch-key species \
  --cell-type-key cell_type_ontology_term_id \
  --finetune-max-len 2200 \
  --embed-max-len 2800 \
  --num-epochs 8 \
  --mmd-target species \
  --mmd-scale 0.03 \
  --include-organism-classification \
  --align-checkpoint-genes \
  --use-knn \
  --rebuild-knn-graph \
  --include-unknown-train-labels \
  --seed 42 \
  --notebook-provenance \
  "Pandora fresh rerun matched to task3: small-v2; reconstruction+cell type+organism classification+KL; differentiable species MMD=0.03 on cell-type token; forced CP10K/log1p/PCA/neighbors; multi-cell kNN"

srun "$PYTHON" scripts/build_task3_token_concat_pca.py \
  --input "$ZERO_SHOT" \
  --output "$ZERO_SHOT_CONCAT" \
  --method "scPRINT-2 zero-shot all tokens except assay and organism PCA50" \
  --token-policy all-except-assay-organism \
  --seed 42

srun "$PYTHON" scripts/build_task3_token_concat_pca.py \
  --input "$FINE_TUNED" \
  --output "$FINE_TUNED_CONCAT" \
  --method "scPRINT-2 fine-tuned all tokens except assay and organism PCA50" \
  --token-policy all-except-assay-organism \
  --seed 42

srun "$PYTHON" scripts/build_expression_scib_baselines.py \
  --input "$SOURCE" \
  --output "$BASELINES" \
  --seed 42

"$PYTHON" - "$ROOT" "$SOURCE" "$CHECKPOINT" "$ZERO_SHOT" "$FINE_TUNED" "$ZERO_SHOT_CONCAT" "$FINE_TUNED_CONCAT" "$BASELINES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


root = Path(sys.argv[1])
files = [Path(value) for value in sys.argv[2:]]
payload = {
    "run_id": root.name,
    "task3_match": {
        "checkpoint": "small-v2.ckpt",
        "seed": 42,
        "cell_type_key": "cell_type_ontology_term_id",
        "batch_key": "species",
        "knn": "normalize_total(1e4) -> log1p -> PCA50 -> neighbors",
        "finetune_max_len": 2200,
        "embed_max_len": 2800,
        "num_epochs": 8,
        "mmd": "differentiable balanced species-pair MMD on cell-type token",
        "mmd_scale": 0.03,
        "organism_classification": True,
        "restore_best_model": True,
    },
    "fresh_outputs": {
        path.name: {"path": str(path.resolve()), "sha256": digest(path)}
        for path in files
    },
}
(root / "fresh_scprint2_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
PY
touch "$ROOT/SCPRINT2_EMBEDDINGS_COMPLETE"
