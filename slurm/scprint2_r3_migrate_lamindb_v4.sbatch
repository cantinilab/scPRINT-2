#!/bin/bash
#SBATCH --job-name=scprint2-r3-db-migrate
#SBATCH --partition=compil
#SBATCH --account=wbg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-db-migrate-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-db-migrate-%j.err
set -euo pipefail

R=/lustre/fswork/projects/rech/xeg/uat95fg
E=$R/scprint2_basev3_revision3_t_6cff2510
P=$R/scPRINT-esi-primary-t_6cff2510
S=$R/scDataLoader-esi-primary-t_6cff2510
RUNTIME=$R/scprint2_runtime_eb6f951_t_6cff2510
HELPER=$E/migrate_lamindb2_offline.py
ORIGINAL=$R/scprint2_base_v3_filt_esi_t_6cff2510.lndb
CANDIDATE=$R/scprint2_base_v3_filt_esi_t_6cff2510_lamindb2_candidate_v4.lndb
RECEIPT=$E/lamindb2_migration_candidate_v4.json
COMMIT=eb6f951f6b9409159252816f8d9db09094615551
SCD=0a2fb2080dd2bf471c1532c04ee0789ce06441c2
ORIGINAL_SHA=cec47585eb6ebcfebe65f00e3c085df5b6b518c476cb305b3cb55bf387a16389
HELPER_SHA=ee54ea5209ac55efb175bc0d302afb4acf0bfd716467767eea0b3f2b0036b181

[[ $(git -C "$P" rev-parse HEAD) == "$COMMIT" ]]
[[ $(git -C "$S" rev-parse HEAD) == "$SCD" ]]
[[ -z $(git -C "$P" status --porcelain) ]]
[[ -z $(git -C "$S" status --porcelain) ]]
[[ -s $ORIGINAL && -s $HELPER ]]
[[ ! -e $CANDIDATE && ! -e $RECEIPT ]]
[[ $(sha256sum "$HELPER" | cut -d' ' -f1) == "$HELPER_SHA" ]]
python3 "$P/scripts/training/verify_frozen_runtime.py" \
  verify "$RUNTIME" "$P" "$S" "$COMMIT" "$SCD"

printf 'LAMINDB2_OFFLINE_MIGRATION_START %s original=%s candidate=%s helper_sha=%s network=none\n' \
  "$(date -Is)" "$ORIGINAL" "$CANDIDATE" "$HELPER_SHA"
"$RUNTIME/.venv/bin/python" "$HELPER" \
  "$ORIGINAL" "$CANDIDATE" "$RECEIPT" "$ORIGINAL_SHA"
printf 'LAMINDB2_OFFLINE_MIGRATION_JOB_PASS %s receipt=%s\n' "$(date -Is)" "$RECEIPT"
