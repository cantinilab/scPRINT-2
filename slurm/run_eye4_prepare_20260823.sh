#!/bin/bash
#SBATCH --job-name=eye4-prep
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_matched_20260823/logs/prepare-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_matched_20260823/logs/prepare-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_matched_20260823}"
SCRIPT="${REPO_ROOT}/scripts/prepare_eye_cross_species.py"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${ROOT}/tmp"
mkdir -p "${ROOT}/geo" "${ROOT}/species" "$TMPDIR"

verify_download() {
  local accession="$1" filename="$2"
  local output="${ROOT}/geo/${filename}"
  [[ -s "$output" ]] || {
    echo "Missing ${output}; GEO downloads must be performed on the submit node" >&2
    return 1
  }
  gzip -t "$output"
  echo "VERIFIED_GEO ${accession} $(stat -c %s "$output") ${output}"
}

verify_download GSE148371 GSE148371_Human_count_matrix.csv.gz
verify_download GSE148373 GSE148373_MacaF_count_matrix.csv.gz
verify_download GSE146186 GSE146186_Mouse_count_matrix.csv.gz
verify_download GSE146187 GSE146187_Pig_count_matrix.csv.gz

if [[ ! -s "${ROOT}/species/human.h5ad" ]]; then
  srun "$PYTHON" "$SCRIPT" build-species \
    --counts "${ROOT}/geo/GSE148371_Human_count_matrix.csv.gz" \
    --annotations "${ROOT}/annotations/human.tsv" \
    --mapping "${ROOT}/orthologs/human_to_mouse_mm10.csv" \
    --mapping "${ROOT}/orthologs/human_mouse.csv" \
    --source-column human_gene \
    --output "${ROOT}/species/human.h5ad"
fi

if [[ ! -s "${ROOT}/species/macaque_fascicularis.h5ad" ]]; then
  srun "$PYTHON" "$SCRIPT" build-species \
    --counts "${ROOT}/geo/GSE148373_MacaF_count_matrix.csv.gz" \
    --annotations "${ROOT}/annotations/macaque_fascicularis.tsv" \
    --mapping "${ROOT}/orthologs/mouse_to_MF.csv" \
    --source-column MF_gene \
    --output "${ROOT}/species/macaque_fascicularis.h5ad"
fi

if [[ ! -s "${ROOT}/species/mouse.h5ad" ]]; then
  srun "$PYTHON" "$SCRIPT" build-species \
    --counts "${ROOT}/geo/GSE146186_Mouse_count_matrix.csv.gz" \
    --annotations "${ROOT}/annotations/mouse.tsv" \
    --identity-mapping \
    --output "${ROOT}/species/mouse.h5ad"
fi

if [[ ! -s "${ROOT}/species/pig.h5ad" ]]; then
  srun "$PYTHON" "$SCRIPT" build-species \
    --counts "${ROOT}/geo/GSE146187_Pig_count_matrix.csv.gz" \
    --annotations "${ROOT}/annotations/pig.tsv" \
    --mapping "${ROOT}/orthologs/mouse_to_pig.csv" \
    --source-column pig_gene \
    --output "${ROOT}/species/pig.h5ad"
fi

if [[ ! -s "${ROOT}/eye4_orthology_audit.tsv" ]]; then
  srun "$PYTHON" "$SCRIPT" write-orthology-audit \
    --input "${ROOT}/species/mouse.h5ad" \
    --input "${ROOT}/species/human.h5ad" \
    --input "${ROOT}/species/macaque_fascicularis.h5ad" \
    --input "${ROOT}/species/pig.h5ad" \
    --output "${ROOT}/eye4_orthology_audit.tsv"
fi

RAW="${ROOT}/eye4_mouse_one2one_raw_counts.h5ad"
if [[ ! -s "$RAW" ]]; then
  srun "$PYTHON" "$SCRIPT" combine \
    --input "${ROOT}/species/mouse.h5ad" \
    --input "${ROOT}/species/human.h5ad" \
    --input "${ROOT}/species/macaque_fascicularis.h5ad" \
    --input "${ROOT}/species/pig.h5ad" \
    --min-shared-genes 900 \
    --output "$RAW"
fi

PREPARED="${ROOT}/eye4_mouse_one2one_scprint_preprocessed.h5ad"
REFERENCE_GENES="${WORK_ROOT}/scPRINT/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad"
if [[ ! -s "$PREPARED" ]]; then
  test -s "$REFERENCE_GENES"
  srun "$PYTHON" "$SCRIPT" preprocess \
    --input "$RAW" \
    --min-valid-genes 500 \
    --min-nnz-genes 200 \
    --reference "$REFERENCE_GENES" \
    --output "$PREPARED" \
    --manifest "${ROOT}/eye4_preparation_manifest.json" \
    --complete "${ROOT}/PREPARE.COMPLETE"
fi

if [[ ! -s "${ROOT}/eye4_preparation_manifest.json" || ! -s "${ROOT}/PREPARE.COMPLETE" ]]; then
  srun "$PYTHON" "$SCRIPT" write-manifest \
    --input "$PREPARED" \
    --manifest "${ROOT}/eye4_preparation_manifest.json" \
    --complete "${ROOT}/PREPARE.COMPLETE"
fi

test -s "$PREPARED"
test -s "${ROOT}/eye4_preparation_manifest.json"
test -s "${ROOT}/PREPARE.COMPLETE"
sha256sum "${ROOT}/geo/"*.csv.gz "$RAW" "$PREPARED" > "${ROOT}/preparation_sha256.tsv"
