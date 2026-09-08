#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=slurm/task3-fresh-ontology-knn-mmd003-%j.out
#SBATCH --error=slurm/task3-fresh-ontology-knn-mmd003-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/12.2.0

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
PYTHON="${SCPRINT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
CHECKPOINT="${SMALL_V2_CHECKPOINT:-${WORK_ROOT}/models/small-v2.ckpt}"
RUN_ID="task3_fresh_ontology_knn_mmd003_20260815"
ROOT="${REPO_ROOT}/data/results/cross_species_embedding/${RUN_ID}"
ZERO_SHOT="${ROOT}/scprint2_zero_shot_small_v2_knn.h5ad"
FINE_TUNED="${ROOT}/scprint2_ft_small_v2_mmd003_knn.h5ad"
ZERO_SHOT_CONCAT="${ROOT}/scprint2_zero_shot_all_tokens_except_assay_organism_pca50.h5ad"

test -s "$SOURCE"
test -s "$CHECKPOINT"
if [[ -e "$ROOT" ]]; then
  echo "Refusing to reuse or overwrite fresh-run root: $ROOT" >&2
  exit 3
fi
mkdir -p "$ROOT"

srun "$PYTHON" "$REPO_ROOT/scripts/run_task3_cross_species_embedding.py" \
  scprint2-zero-shot \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$ZERO_SHOT" \
  --batch-key orig.ident \
  --cell-type-key cell_type_ontology_term_id \
  --embed-how "random expr" \
  --embed-max-len 2800 \
  --use-knn \
  --rebuild-knn-graph \
  --seed 42 \
  --notebook-provenance \
  "fresh task3 zero-shot; small-v2; ontology labels; forced normalize/log1p/PCA/neighbors; multi-cell kNN"

srun "$PYTHON" "$REPO_ROOT/scripts/run_task3_cross_species_embedding.py" \
  scprint2-ft \
  --input "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output "$FINE_TUNED" \
  --batch-key orig.ident \
  --cell-type-key cell_type_ontology_term_id \
  --finetune-max-len 2200 \
  --embed-max-len 2800 \
  --num-epochs 8 \
  --mmd-target species \
  --mmd-scale 0.03 \
  --include-organism-classification \
  --use-knn \
  --rebuild-knn-graph \
  --include-unknown-train-labels \
  --seed 42 \
  --notebook-provenance \
  "fresh task3 fine-tuning; small-v2; differentiable species MMD=0.03 on cell-type token; forced normalize/log1p/PCA/neighbors; multi-cell kNN"

srun "$PYTHON" "$REPO_ROOT/scripts/build_task3_token_concat_pca.py" \
  --input "$ZERO_SHOT" \
  --output "$ZERO_SHOT_CONCAT" \
  --method "scPRINT-2 zero-shot all tokens except assay and organism PCA50" \
  --token-policy all-except-assay-organism \
  --seed 42

"$PYTHON" - "$ROOT" "$SOURCE" "$CHECKPOINT" "$ZERO_SHOT" "$FINE_TUNED" "$ZERO_SHOT_CONCAT" <<'PY'
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
    "fresh_outputs": {
        path.name: {"path": str(path.resolve()), "sha256": digest(path)}
        for path in files
    },
}
(root / "fresh_embedding_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
PY
touch "$ROOT/EMBEDDINGS_COMPLETE"
