#!/bin/bash
#SBATCH --job-name=scprint2-r3-db-verify
#SBATCH --partition=prepost
#SBATCH --account=wbg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-db-verify-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-db-verify-%j.err
set -euo pipefail

R=/lustre/fswork/projects/rech/xeg/uat95fg
P=$R/scPRINT-esi-primary-t_6cff2510
S=$R/scDataLoader-esi-primary-t_6cff2510
E=$R/scprint2_basev3_revision3_t_6cff2510
ORIGINAL_DB=${ORIGINAL_DB:?submitter must bind immutable legacy SQLite}
CANDIDATE_DB=${CANDIDATE_DB:?submitter must bind unswapped migrated candidate}
MIGRATION_RECEIPT=${MIGRATION_RECEIPT:?submitter must bind migration receipt}
VERIFICATION_RECEIPT=${VERIFICATION_RECEIPT:?submitter must bind new absent verification receipt}
EXPECTED_SCPRINT_COMMIT=${EXPECTED_SCPRINT_COMMIT:?submitter must bind reviewed PR head}
EXPECTED_SCDATALOADER_COMMIT=0a2fb2080dd2bf471c1532c04ee0789ce06441c2
REGISTRY_MANIFEST=$E/primary_filtered_registry_manifest.json
MEMBERSHIP_CONTRACT=$P/config/base_v3_ntv3_primary_membership.json

[[ $(git -C "$P" rev-parse HEAD) == "$EXPECTED_SCPRINT_COMMIT" ]]
[[ $(git -C "$S" rev-parse HEAD) == "$EXPECTED_SCDATALOADER_COMMIT" ]]
[[ -z $(git -C "$P" status --porcelain) ]]
[[ -z $(git -C "$S" status --porcelain) ]]
[[ -s $ORIGINAL_DB && -s $CANDIDATE_DB && -s $MIGRATION_RECEIPT ]]
[[ -s $REGISTRY_MANIFEST && -s $MEMBERSHIP_CONTRACT ]]
[[ ! -e $VERIFICATION_RECEIPT ]]

printf 'LAMINDB2_INDEPENDENT_VERIFY_START %s commit=%s candidate=%s\n' \
  "$(date -Is)" "$EXPECTED_SCPRINT_COMMIT" "$CANDIDATE_DB"
python3 "$P/scripts/training/verify_migrated_lamindb.py" create \
  "$ORIGINAL_DB" "$CANDIDATE_DB" "$MIGRATION_RECEIPT" \
  "$REGISTRY_MANIFEST" "$MEMBERSHIP_CONTRACT" "$VERIFICATION_RECEIPT"
python3 "$P/scripts/training/verify_migrated_lamindb.py" verify \
  "$VERIFICATION_RECEIPT" "$CANDIDATE_DB"
printf 'LAMINDB2_INDEPENDENT_VERIFY_PASS %s receipt=%s\n' \
  "$(date -Is)" "$VERIFICATION_RECEIPT"
