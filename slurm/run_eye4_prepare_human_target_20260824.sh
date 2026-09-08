#!/bin/bash
#SBATCH --job-name=eye4-human-prep
#SBATCH --account=xeg@h100
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/prepare-human-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_human_target_hybrid_20260824/logs/prepare-human-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u

WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT-eye-20260823}"
PYTHON="${SCPRINT_PYTHON:-${WORK_ROOT}/scPRINT/.venv/bin/python}"
SOURCE_ROOT="${EYE4_SOURCE_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_task3_corrected_20260823}"
ROOT="${EYE4_ROOT:-${WORK_ROOT}/data/eye_cross_species/eye4_human_target_hybrid_20260824}"
SCRIPT="${REPO_ROOT}/scripts/prepare_eye_cross_species.py"
CHECKPOINT="${SMALL_V2_CHECKPOINT:-${WORK_ROOT}/models/small-v2.ckpt}"
MAPPINGS="${SOURCE_ROOT}/orthologs/biomart_corrected"
HUMAN_ORTHOLOGS="${SOURCE_ROOT}/orthologs/biomart_human_target_20260824"
HUMAN_TAXON="NCBITaxon:9606"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${ROOT}/tmp"
mkdir -p "${ROOT}/annotations" "${ROOT}/species/human_target" \
  "${ROOT}/species/checkpoint_human" "${ROOT}/logs" "$TMPDIR"

test -s "$CHECKPOINT"
test -s "${MAPPINGS}/human_native_genes_biomart.tsv"
test -s "${MAPPINGS}/human_to_mouse_biomart.tsv"
test -s "${HUMAN_ORTHOLOGS}/pig_to_human_biomart.tsv"

rewrite_annotation() {
  local species="$1" source_taxon="$2"
  local output="${ROOT}/annotations/${species}.tsv"
  if [[ -s "$output" ]]; then
    return
  fi
  srun "$PYTHON" "$SCRIPT" rewrite-annotations \
    --input "${SOURCE_ROOT}/annotations/${species}.tsv" \
    --species "$species" \
    --source-taxon "$source_taxon" \
    --feature-taxon "$HUMAN_TAXON" \
    --output "$output"
}

rewrite_annotation human NCBITaxon:9606
rewrite_annotation macaque_fascicularis NCBITaxon:9541
rewrite_annotation mouse NCBITaxon:10090
rewrite_annotation pig NCBITaxon:9823

build_direct_human_species() {
  local species="$1" counts="$2"
  local output="${ROOT}/species/human_target/${species}.h5ad"
  local suffix_args=()
  if [[ "$species" == "macaque_fascicularis" ]]; then
    suffix_args+=(--normalize-terminal-np-suffixes)
  fi
  if [[ -s "$output" ]]; then
    return
  fi
  srun "$PYTHON" "$SCRIPT" build-species \
    --counts "${SOURCE_ROOT}/geo/${counts}" \
    --annotations "${ROOT}/annotations/${species}.tsv" \
    --mapping "${MAPPINGS}/human_native_genes_biomart.tsv" \
    --source-column source_gene_symbol \
    --target-column source_ensembl_gene_id \
    --mapping-mode direct-human-symbol \
    --target-id-type ensembl \
    "${suffix_args[@]}" \
    --output "$output"
}

build_direct_human_species human GSE148371_Human_count_matrix.csv.gz
build_direct_human_species macaque_fascicularis GSE148373_MacaF_count_matrix.csv.gz

PIG_OUTPUT="${ROOT}/species/human_target/pig.h5ad"
if [[ ! -s "$PIG_OUTPUT" ]]; then
  srun "$PYTHON" "$SCRIPT" build-species \
    --counts "${SOURCE_ROOT}/geo/GSE146187_Pig_count_matrix.csv.gz" \
    --annotations "${ROOT}/annotations/pig.tsv" \
    --direct-registry "${MAPPINGS}/human_native_genes_biomart.tsv" \
    --mapping "${HUMAN_ORTHOLOGS}/pig_to_human_biomart.tsv" \
    --source-column source_gene_symbol \
    --source-column source_ensembl_gene_id \
    --target-column human_ensembl_gene_id \
    --mapping-mode direct-human-symbol-with-orthology-fallback \
    --target-id-type ensembl \
    --output "$PIG_OUTPUT"
fi

MOUSE_OUTPUT="${ROOT}/species/human_target/mouse.h5ad"
if [[ ! -s "$MOUSE_OUTPUT" ]]; then
  srun "$PYTHON" "$SCRIPT" build-species \
    --counts "${SOURCE_ROOT}/geo/GSE146186_Mouse_count_matrix.csv.gz" \
    --annotations "${ROOT}/annotations/mouse.tsv" \
    --mapping "${MAPPINGS}/human_to_mouse_biomart.tsv" \
    --source-column mouse_gene_symbol \
    --source-column mouse_ensembl_gene_id \
    --target-column source_ensembl_gene_id \
    --mapping-mode source-unique-target-aggregated-orthology \
    --target-id-type ensembl \
    --output "$MOUSE_OUTPUT"
fi

INPUTS=(
  "${ROOT}/species/human_target/mouse.h5ad"
  "${ROOT}/species/human_target/human.h5ad"
  "${ROOT}/species/human_target/macaque_fascicularis.h5ad"
  "${ROOT}/species/human_target/pig.h5ad"
)

for input in "${INPUTS[@]}"; do
  species="$(basename "${input%.h5ad}")"
  aligned="${ROOT}/species/checkpoint_human/${species}.h5ad"
  if [[ ! -s "$aligned" ]]; then
    srun "$PYTHON" "$SCRIPT" align-checkpoint \
      --input "$input" \
      --checkpoint "$CHECKPOINT" \
      --min-checkpoint-genes 19000 \
    --min-input-gene-overlap 15000 \
      --output "$aligned"
  fi
done

if [[ ! -s "${ROOT}/eye4_human_target_gene_audit.tsv" ]]; then
  srun "$PYTHON" "$SCRIPT" write-orthology-audit \
    --input "${INPUTS[0]}" \
    --input "${INPUTS[1]}" \
    --input "${INPUTS[2]}" \
    --input "${INPUTS[3]}" \
    --output "${ROOT}/eye4_human_target_gene_audit.tsv"
fi

CHECKPOINT_INPUTS=(
  "${ROOT}/species/checkpoint_human/mouse.h5ad"
  "${ROOT}/species/checkpoint_human/human.h5ad"
  "${ROOT}/species/checkpoint_human/macaque_fascicularis.h5ad"
  "${ROOT}/species/checkpoint_human/pig.h5ad"
)
if [[ ! -s "${ROOT}/eye4_human_checkpoint_gene_audit.tsv" ]]; then
  srun "$PYTHON" "$SCRIPT" write-native-gene-audit \
    --input "${CHECKPOINT_INPUTS[0]}" \
    --input "${CHECKPOINT_INPUTS[1]}" \
    --input "${CHECKPOINT_INPUTS[2]}" \
    --input "${CHECKPOINT_INPUTS[3]}" \
    --output "${ROOT}/eye4_human_checkpoint_gene_audit.tsv"
fi

RAW="${ROOT}/eye4_human_ensembl_union_raw_counts.h5ad"
if [[ ! -s "$RAW" ]]; then
  srun "$PYTHON" "$SCRIPT" combine-target-union \
    --input "${INPUTS[0]}" \
    --input "${INPUTS[1]}" \
    --input "${INPUTS[2]}" \
    --input "${INPUTS[3]}" \
    --feature-taxon "$HUMAN_TAXON" \
    --feature-space human_ensembl_union_direct_symbols_species_ortholog_fallback \
    --min-genes-per-species 15000 \
    --output "$RAW"
fi

PREPARED="${ROOT}/eye4_human_ensembl_small_v2.h5ad"
if [[ ! -s "$PREPARED" ]]; then
  srun "$PYTHON" "$SCRIPT" align-checkpoint \
    --input "$RAW" \
    --checkpoint "$CHECKPOINT" \
    --min-checkpoint-genes 19000 \
    --min-input-gene-overlap 18000 \
    --output "$PREPARED"
fi

if [[ ! -s "${ROOT}/eye4_human_preparation_manifest.json" ]]; then
  srun "$PYTHON" "$SCRIPT" write-manifest \
    --input "$PREPARED" \
    --manifest "${ROOT}/eye4_human_preparation_manifest.json" \
    --complete "${ROOT}/HUMAN_TARGET_PREPARE.COMPLETE"
fi

test -s "$PREPARED"
test -s "${ROOT}/eye4_human_target_gene_audit.tsv"
test -s "${ROOT}/eye4_human_checkpoint_gene_audit.tsv"
test -s "${ROOT}/eye4_human_preparation_manifest.json"
test -s "${ROOT}/HUMAN_TARGET_PREPARE.COMPLETE"
sha256sum "${INPUTS[@]}" "$RAW" "$PREPARED" \
  > "${ROOT}/human_target_preparation_sha256.tsv"
