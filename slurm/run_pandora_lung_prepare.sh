#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=slurm/pandora-prepare-%j.out
#SBATCH --error=slurm/pandora-prepare-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
DATA_ROOT="${PANDORA_DATA_ROOT:-${WORK_ROOT}/data/pandora_lung}"
MANIFEST="${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/pandora_lung_atlas_manifest.tsv"
MOUSE_MAPPING="${DATA_ROOT}/mouse_symbol_to_ensembl.tsv"
PYTHON="${REPO_ROOT}/.venv/bin/python"
export TMPDIR="${DATA_ROOT}/tmp"
mkdir -p "${DATA_ROOT}/rds" "${DATA_ROOT}/h5ad" "$TMPDIR"

if [[ ! -s "$MOUSE_MAPPING" ]]; then
  "$PYTHON" "${REPO_ROOT}/scripts/prepare_pandora_lung_atlas.py" download-mapping \
    --target-feature-space mouse_homolog \
    --output "$MOUSE_MAPPING"
fi

download_one() {
  local species="$1" organism="$2" filename="$3" url="$4"
  local output="${DATA_ROOT}/rds/${filename}"
  local expected actual
  if [[ -s "$output" ]]; then
    echo "REUSED ${species} $(stat -c %s "$output") ${output}"
    return 0
  fi
  expected=$(curl --fail --silent --show-error --head --location "$url" \
    | tr -d '\r' | awk 'tolower($1) == "content-length:" {print $2}' | tail -1)
  actual=$(stat -c %s "$output" 2>/dev/null || echo 0)
  if [[ "$actual" != "$expected" ]]; then
    curl --fail --location --retry 12 --retry-all-errors --retry-delay 10 \
      --continue-at - --output "$output" "$url"
  fi
  actual=$(stat -c %s "$output")
  [[ "$actual" == "$expected" ]] || { echo "Size mismatch for ${species}" >&2; return 1; }
  echo "DOWNLOADED ${species} $(stat -c %s "$output") ${output}"
}
export -f download_one
export DATA_ROOT
tail -n +2 "$MANIFEST" | xargs -P 6 -n 4 bash -c 'download_one "$@"' _

inputs=()
while IFS=$'\t' read -r species organism filename url; do
  [[ "$species" == "species" ]] && continue
  output="${DATA_ROOT}/h5ad/${species}.mouse_homolog_ensmusg.h5ad"
  if [[ ! -s "$output" ]]; then
    Rscript "${REPO_ROOT}/scripts/convert_pandora_lung_rds.R" \
      "${DATA_ROOT}/rds/${filename}" "$MOUSE_MAPPING" "$output" \
      "$organism" "$species" "$PYTHON" mouse_homolog
  fi
  inputs+=(--input "$output")
done < "$MANIFEST"

combined="${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_full.h5ad"
"$PYTHON" "${REPO_ROOT}/scripts/prepare_pandora_lung_atlas.py" combine \
  "${inputs[@]}" --join outer --output "$combined"
filtered="${DATA_ROOT}/pandora_lung_mouse_homolog_ensmusg_filtered_93423.h5ad"
"$PYTHON" "${REPO_ROOT}/scripts/prepare_pandora_lung_atlas.py" filter-labels \
  --input "$combined" \
  --output "$filtered" \
  --label-key cell_type_ontology_term_id \
  --exclude-label unknown \
  --exclude-label unmapped \
  --exclude-label mix \
  --expected-cells 93423
"$PYTHON" - "$combined" "${DATA_ROOT}/annotation_audit.tsv" "$filtered" "${DATA_ROOT}/annotation_audit_filtered_93423.tsv" <<'PY'
import sys
import anndata as ad

for input_path, output_path in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    adata = ad.read_h5ad(input_path, backed="r")
    table = (
        adata.obs.groupby(["species", "cell_type"], observed=True)
        .size()
        .rename("cells")
        .reset_index()
    )
    table.to_csv(output_path, sep="\t", index=False)
    print(input_path)
    print(adata)
    print(table.to_string(index=False))
PY
touch "${DATA_ROOT}/PREPARE_MOUSE_HOMOLOG_ENSMUSG.COMPLETE"
