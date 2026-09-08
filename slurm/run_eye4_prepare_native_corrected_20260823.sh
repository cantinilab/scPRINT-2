#!/bin/bash
#SBATCH --job-name=eye4-native-prep
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/prepare-native-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823/logs/prepare-native-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_corrected_20260823}"
REJECTED_ROOT="${WORK_ROOT}/data/eye_cross_species/eye4_task3_matched_20260823"
SCRIPT="${REPO_ROOT}/scripts/prepare_eye_cross_species.py"
CHECKPOINT="${SMALL_V2_CHECKPOINT:-${WORK_ROOT}/models/small-v2.ckpt}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${ROOT}/tmp"
mkdir -p "${ROOT}/geo" "${ROOT}/annotations" "${ROOT}/species/raw" \
  "${ROOT}/species/native_mapped" "${ROOT}/species/checkpoint" "$TMPDIR"

for source in "${REJECTED_ROOT}/geo/"*.csv.gz; do
  target="${ROOT}/geo/$(basename "$source")"
  if [[ ! -e "$target" ]]; then
    ln -s "$source" "$target"
  fi
  gzip -t "$target"
done

rewrite_annotations() {
  local species="$1" source_taxon="$2" feature_taxon="$3"
  local input="${REJECTED_ROOT}/annotations/${species}.tsv"
  local output="${ROOT}/annotations/${species}.tsv"
  if [[ ! -s "$output" ]]; then
    srun "$PYTHON" "$SCRIPT" rewrite-annotations \
      --input "$input" \
      --species "$species" \
      --source-taxon "$source_taxon" \
      --feature-taxon "$feature_taxon" \
      --output "$output"
  fi
}

rewrite_annotations human NCBITaxon:9606 NCBITaxon:9606
rewrite_annotations macaque_fascicularis NCBITaxon:9541 NCBITaxon:9544
rewrite_annotations mouse NCBITaxon:10090 NCBITaxon:10090
rewrite_annotations pig NCBITaxon:9823 NCBITaxon:9823

build_native_species() {
  local species="$1" counts="$2"
  local mapping="${ROOT}/orthologs/biomart_corrected/${species}_native_genes_biomart.tsv"
  local mapped="${ROOT}/species/native_mapped/${species}.h5ad"
  local checkpoint_aligned="${ROOT}/species/checkpoint/${species}.h5ad"
  test -s "$mapping"
  if [[ ! -s "$mapped" ]]; then
    srun "$PYTHON" "$SCRIPT" build-species \
      --counts "${ROOT}/geo/${counts}" \
      --annotations "${ROOT}/annotations/${species}.tsv" \
      --mapping "$mapping" \
      --source-column source_gene_symbol \
      --source-column source_ensembl_gene_id \
      --target-column source_ensembl_gene_id \
      --output "$mapped"
  fi
  if [[ ! -s "$checkpoint_aligned" ]]; then
    srun "$PYTHON" "$SCRIPT" align-checkpoint \
      --input "$mapped" \
      --checkpoint "$CHECKPOINT" \
      --min-checkpoint-genes 15000 \
      --min-input-gene-overlap 10000 \
      --output "$checkpoint_aligned"
  fi
}

build_native_species human GSE148371_Human_count_matrix.csv.gz
build_native_species macaque_fascicularis GSE148373_MacaF_count_matrix.csv.gz
build_native_species mouse GSE146186_Mouse_count_matrix.csv.gz
build_native_species pig GSE146187_Pig_count_matrix.csv.gz

CHECKPOINT_INPUTS=(
  "${ROOT}/species/checkpoint/mouse.h5ad"
  "${ROOT}/species/checkpoint/human.h5ad"
  "${ROOT}/species/checkpoint/macaque_fascicularis.h5ad"
  "${ROOT}/species/checkpoint/pig.h5ad"
)

if [[ ! -s "${ROOT}/eye4_native_gene_audit.tsv" ]]; then
  srun "$PYTHON" "$SCRIPT" write-native-gene-audit \
    --input "${CHECKPOINT_INPUTS[0]}" \
    --input "${CHECKPOINT_INPUTS[1]}" \
    --input "${CHECKPOINT_INPUTS[2]}" \
    --input "${CHECKPOINT_INPUTS[3]}" \
    --output "${ROOT}/eye4_native_gene_audit.tsv"
fi

PREPARED="${ROOT}/eye4_native_scprint_preprocessed.h5ad"
if [[ ! -s "$PREPARED" ]]; then
  srun "$PYTHON" "$SCRIPT" combine-native \
    --input "${CHECKPOINT_INPUTS[0]}" \
    --input "${CHECKPOINT_INPUTS[1]}" \
    --input "${CHECKPOINT_INPUTS[2]}" \
    --input "${CHECKPOINT_INPUTS[3]}" \
    --checkpoint "$CHECKPOINT" \
    --min-genes-per-species 15000 \
    --output "$PREPARED"
fi

if [[ ! -s "${ROOT}/eye4_preparation_manifest.json" ]]; then
  srun "$PYTHON" "$SCRIPT" write-manifest \
    --input "$PREPARED" \
    --manifest "${ROOT}/eye4_preparation_manifest.json" \
    --complete "${ROOT}/PREPARE.COMPLETE"
fi

test -s "$PREPARED"
test -s "${ROOT}/eye4_native_gene_audit.tsv"
test -s "${ROOT}/eye4_preparation_manifest.json"
test -s "${ROOT}/PREPARE.COMPLETE"
sha256sum "${ROOT}/geo/"*.csv.gz "${CHECKPOINT_INPUTS[@]}" "$PREPARED" \
  > "${ROOT}/preparation_sha256.tsv"
