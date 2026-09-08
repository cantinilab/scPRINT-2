#!/bin/bash
#SBATCH --job-name=eye4-ortholog-prep
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/prepare-ortholog-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/prepare-ortholog-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_corrected_20260823}"
SCRIPT="${REPO_ROOT}/scripts/prepare_eye_cross_species.py"
REFERENCE="${WORK_ROOT}/scPRINT/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad"
MAPPINGS="${ROOT}/orthologs/biomart_corrected"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${ROOT}/tmp"
mkdir -p "${ROOT}/species/ortholog" "$TMPDIR"

test -s "${MAPPINGS}/BIOMART.COMPLETE"
test -s "$REFERENCE"

build_ortholog_species() {
  local species="$1" counts="$2"
  local output="${ROOT}/species/ortholog/${species}.h5ad"
  if [[ -s "$output" ]]; then
    return
  fi
  if [[ "$species" == "mouse" ]]; then
    srun "$PYTHON" "$SCRIPT" build-species \
      --counts "${ROOT}/geo/${counts}" \
      --annotations "${ROOT}/annotations/${species}.tsv" \
      --identity-mapping \
      --output "$output"
  else
    srun "$PYTHON" "$SCRIPT" build-species \
      --counts "${ROOT}/geo/${counts}" \
      --annotations "${ROOT}/annotations/${species}.tsv" \
      --mapping "${MAPPINGS}/${species}_to_mouse_biomart.tsv" \
      --source-column source_gene_symbol \
      --source-column source_ensembl_gene_id \
      --target-column mouse_gene_symbol \
      --resolve-orthology-conflicts \
      --output "$output"
  fi
}

build_ortholog_species human GSE148371_Human_count_matrix.csv.gz
build_ortholog_species macaque_fascicularis GSE148373_MacaF_count_matrix.csv.gz
build_ortholog_species mouse GSE146186_Mouse_count_matrix.csv.gz
build_ortholog_species pig GSE146187_Pig_count_matrix.csv.gz

INPUTS=(
  "${ROOT}/species/ortholog/mouse.h5ad"
  "${ROOT}/species/ortholog/human.h5ad"
  "${ROOT}/species/ortholog/macaque_fascicularis.h5ad"
  "${ROOT}/species/ortholog/pig.h5ad"
)

if [[ ! -s "${ROOT}/eye4_orthology_audit.tsv" ]]; then
  srun "$PYTHON" "$SCRIPT" write-orthology-audit \
    --input "${INPUTS[0]}" \
    --input "${INPUTS[1]}" \
    --input "${INPUTS[2]}" \
    --input "${INPUTS[3]}" \
    --output "${ROOT}/eye4_orthology_audit.tsv"
fi

RAW="${ROOT}/eye4_mouse_ortholog_union_raw_counts.h5ad"
if [[ ! -s "$RAW" ]]; then
  srun "$PYTHON" "$SCRIPT" combine-ortholog-union \
    --input "${INPUTS[0]}" \
    --input "${INPUTS[1]}" \
    --input "${INPUTS[2]}" \
    --input "${INPUTS[3]}" \
    --min-genes-per-species 10000 \
    --min-common-genes 8000 \
    --output "$RAW"
fi

PREPARED="${ROOT}/eye4_mouse_ortholog_scprint_preprocessed.h5ad"
if [[ ! -s "$PREPARED" ]]; then
  srun "$PYTHON" "$SCRIPT" preprocess \
    --input "$RAW" \
    --min-valid-genes 10000 \
    --min-nnz-genes 200 \
    --reference "$REFERENCE" \
    --output "$PREPARED" \
    --manifest "${ROOT}/eye4_ortholog_preparation_manifest.json" \
    --complete "${ROOT}/ORTHOLOG_PREPARE.COMPLETE"
fi

test -s "$PREPARED"
test -s "${ROOT}/eye4_orthology_audit.tsv"
test -s "${ROOT}/eye4_ortholog_preparation_manifest.json"
test -s "${ROOT}/ORTHOLOG_PREPARE.COMPLETE"
sha256sum "${INPUTS[@]}" "$RAW" "$PREPARED" \
  > "${ROOT}/ortholog_preparation_sha256.tsv"
