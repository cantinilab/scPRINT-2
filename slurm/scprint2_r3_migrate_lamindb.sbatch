#!/bin/bash
#SBATCH --job-name=scprint2-r3-db-migrate
#SBATCH --partition=prepost
#SBATCH --account=wbg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-db-migrate-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-db-migrate-%j.err
set -euo pipefail

R=/lustre/fswork/projects/rech/xeg/uat95fg
P=$R/scPRINT-esi-primary-t_6cff2510
S=$R/scDataLoader-esi-primary-t_6cff2510
E=$R/scprint2_basev3_revision3_t_6cff2510
RUNTIME_ROOT=${RUNTIME_ROOT:?submitter must bind accepted exact-head runtime}
ORIGINAL_DB=${ORIGINAL_DB:?submitter must bind immutable legacy SQLite}
CANDIDATE_DB=${CANDIDATE_DB:?submitter must bind new absent candidate}
MIGRATION_RECEIPT=${MIGRATION_RECEIPT:?submitter must bind new absent receipt}
EXPECTED_ORIGINAL_SHA256=${EXPECTED_ORIGINAL_SHA256:?submitter must bind original bytes}
EXPECTED_SCPRINT_COMMIT=${EXPECTED_SCPRINT_COMMIT:?submitter must bind reviewed PR head}
EXPECTED_SCDATALOADER_COMMIT=0a2fb2080dd2bf471c1532c04ee0789ce06441c2
REGISTRY_MANIFEST=$E/primary_filtered_registry_manifest.json
MEMBERSHIP_CONTRACT=$P/config/base_v3_ntv3_primary_membership.json

[[ $(git -C "$P" rev-parse HEAD) == "$EXPECTED_SCPRINT_COMMIT" ]]
[[ $(git -C "$S" rev-parse HEAD) == "$EXPECTED_SCDATALOADER_COMMIT" ]]
[[ -z $(git -C "$P" status --porcelain) ]]
[[ -z $(git -C "$S" status --porcelain) ]]
[[ -s $ORIGINAL_DB && -s $REGISTRY_MANIFEST && -s $MEMBERSHIP_CONTRACT ]]
[[ ! -e $CANDIDATE_DB && ! -e $MIGRATION_RECEIPT ]]
python3 "$P/scripts/training/verify_frozen_runtime.py" verify \
  "$RUNTIME_ROOT" "$P" "$S" "$EXPECTED_SCPRINT_COMMIT" "$EXPECTED_SCDATALOADER_COMMIT"

printf 'LAMINDB2_HARDENED_MIGRATION_START %s commit=%s original=%s candidate=%s\n' \
  "$(date -Is)" "$EXPECTED_SCPRINT_COMMIT" "$ORIGINAL_DB" "$CANDIDATE_DB"
"$RUNTIME_ROOT/.venv/bin/python" "$P/scripts/training/migrate_lamindb2_offline.py" \
  "$ORIGINAL_DB" "$CANDIDATE_DB" "$MIGRATION_RECEIPT" \
  "$EXPECTED_ORIGINAL_SHA256" "$REGISTRY_MANIFEST" "$MEMBERSHIP_CONTRACT"
printf 'LAMINDB2_HARDENED_MIGRATION_PASS %s receipt=%s\n' \
  "$(date -Is)" "$MIGRATION_RECEIPT"
