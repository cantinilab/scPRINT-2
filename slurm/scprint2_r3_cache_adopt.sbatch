#!/bin/bash
#SBATCH --job-name=scprint2-r3-cache-adopt
#SBATCH --partition=prepost
#SBATCH --account=wbg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-cache-adopt-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-cache-adopt-%j.err
set -euo pipefail
R=/lustre/fswork/projects/rech/xeg/uat95fg
P=$R/scPRINT-esi-primary-t_6cff2510
S=$R/scDataLoader-esi-primary-t_6cff2510
ENV=$R/scPRINT/.venv
E=$R/scprint2_basev3_revision3_t_6cff2510
CACHE=/lustre/fsn1/projects/rech/xeg/uat95fg/base_v3_filt_esi_t_6cff2510
DB=$R/scprint2_base_v3_filt_esi_t_6cff2510.lndb
NTV3=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_inputs/t_2bbe0ae6/ntv3-full/gene_embeddings.parquet
REGISTRY_MANIFEST=$E/primary_filtered_registry_manifest.json
RECEIPT=$E/primary_filtered_cache_manifest.json
EXPECTED_SCPRINT_COMMIT=${EXPECTED_SCPRINT_COMMIT:?submitter must bind the reviewed PR head}
EXPECTED_SCDATALOADER_COMMIT=0a2fb2080dd2bf471c1532c04ee0789ce06441c2
test "$(git -C "$P" rev-parse HEAD)" = "$EXPECTED_SCPRINT_COMMIT"
test "$(git -C "$S" rev-parse HEAD)" = "$EXPECTED_SCDATALOADER_COMMIT"
test -z "$(git -C "$P" status --porcelain)"; test -z "$(git -C "$S" status --porcelain)"
test ! -e "$RECEIPT"
for name in categories idx_full.npy klass_indices.pt klass_indices_offsets.pt nnz.npy test_datasets.txt test_idx.npy train_labels.npy valid_idx.npy; do test -s "$CACHE/$name"; done
export PYTHONPATH="$P/scripts/training:$P:$S:$R" LAMIN_CURRENT_INSTANCE=jkobject/scprint2 LAMINDB_DJANGO_DATABASE_URL="sqlite:///$DB"
unset LAMINDB_INSTANCE_DB
export NUMBA_CACHE_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/numba XDG_CACHE_HOME=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/xdg MPLCONFIGDIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/matplotlib
mkdir -p "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
printf 'PRIMARY_CACHE_ADOPTION_START %s commit=%s scd=%s existing_files=9 gpu=none\n' "$(date -Is)" "$EXPECTED_SCPRINT_COMMIT" "$EXPECTED_SCDATALOADER_COMMIT"
"$ENV/bin/python" "$P/scripts/training/adopt_primary_filtered_cache.py" "$P/config/base_v3_ntv3_primary.yml" "$P/config/base_v3_ntv3_primary_membership.json" "$CACHE" "$RECEIPT" "$NTV3" "$REGISTRY_MANIFEST"
printf 'PRIMARY_CACHE_ADOPTION_PASS %s receipt=%s\n' "$(date -Is)" "$RECEIPT"
