#!/bin/bash
#SBATCH --job-name=scprint2-r3-primary-long
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-primary-long-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-primary-long-%j.err
#SBATCH --open-mode=append
#SBATCH --account=wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=22
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --signal=SIGUSR1@600
#SBATCH --requeue
set -euo pipefail
R=/lustre/fswork/projects/rech/xeg/uat95fg
P=$R/scPRINT-esi-primary-t_6cff2510
S=$R/scDataLoader-esi-primary-t_6cff2510
E=$R/scprint2_basev3_revision3_t_6cff2510
RUNTIME_ROOT=${RUNTIME_ROOT:?submitter must bind an accepted frozen runtime}
RUNTIME_MANIFEST=$RUNTIME_ROOT/runtime_manifest.json
ENV=$RUNTIME_ROOT/.venv
CONTROL=$P/scripts/training
DB=${LAMINDB_DATABASE_PATH:?submitter must bind the accepted migrated SQLite}
DB_MIGRATION_RECEIPT=${LAMINDB_MIGRATION_RECEIPT:?submitter must bind its accepted migration receipt}
CACHE=/lustre/fsn1/projects/rech/xeg/uat95fg/base_v3_filt_esi_t_6cff2510
NTV3=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_inputs/t_2bbe0ae6/ntv3-full/gene_embeddings.parquet
GENE_LOCS=$R/scPRINT/data/main/gene_locs/all_locs.parquet
CACHE_RECEIPT=$E/primary_filtered_cache_manifest.json
RUNROOT=$R/scprint2_primary_esi_ntv3_t_6cff2510
ACTIVE=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runs/primary-esi-ntv3-r3-active.json
GROUP=primary-esi-ntv3-t_6cff2510
SEGMENT=${SLURM_RESTART_COUNT:-0}
EXPECTED_SCPRINT_COMMIT=${EXPECTED_SCPRINT_COMMIT:?submitter must bind the reviewed PR head}
EXPECTED_SCDATALOADER_COMMIT=0a2fb2080dd2bf471c1532c04ee0789ce06441c2
export EXPECTED_SCPRINT_COMMIT EXPECTED_SCDATALOADER_COMMIT
SEGMENT_OPEN=0
WANDB_MONITOR_PID=
EXIT_REASON=wrapper-exit
TMPBASE=${JOBSCRATCH:-/tmp}/${SLURM_JOB_ID}-segment-${SEGMENT}-wandb
cleanup(){ rc=$?; if test -n "${WANDB_MONITOR_PID:-}"; then kill "$WANDB_MONITOR_PID" 2>/dev/null || true; wait "$WANDB_MONITOR_PID" 2>/dev/null || true; fi; if test "$SEGMENT_OPEN" -eq 1; then "$ENV/bin/python" "$CONTROL/manage_primary_lineage.py" end "$RUNROOT" "$SLURM_JOB_ID" "$SEGMENT" "$EXIT_REASON" || true; fi; rm -rf "$TMPBASE"; printf 'PRIMARY_LONG_EXIT %s job=%s segment=%s rc=%s reason=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$SEGMENT" "$rc" "$EXIT_REASON"; }
trap cleanup EXIT
if test "$SEGMENT" -ge 5; then "$ENV/bin/python" "$CONTROL/manage_primary_lineage.py" finalize "$RUNROOT" "$ACTIVE"; exit 0; fi
for path in "$P" "$S" "$DB" "$CACHE_RECEIPT" "$CACHE/klass_indices.pt" "$CACHE/train_weights.npy" "$NTV3" "$GENE_LOCS"; do test -e "$path"; done
test "$(git -C "$P" rev-parse HEAD)" = "$EXPECTED_SCPRINT_COMMIT"
test "$(git -C "$S" rev-parse HEAD)" = "$EXPECTED_SCDATALOADER_COMMIT"
test -z "$(git -C "$P" status --porcelain)"; test -z "$(git -C "$S" status --porcelain)"
test -s "$RUNTIME_MANIFEST"
python3 "$CONTROL/verify_frozen_runtime.py" verify "$RUNTIME_ROOT" "$P" "$S" "$EXPECTED_SCPRINT_COMMIT" "$EXPECTED_SCDATALOADER_COMMIT"
"$ENV/bin/python" "$CONTROL/verify_migrated_lamindb.py" "$DB_MIGRATION_RECEIPT" "$DB"
export PYTHONPATH="$CONTROL:$P:$S:$R" LAMIN_CURRENT_INSTANCE=jkobject/scprint2 LAMINDB_DJANGO_DATABASE_URL="sqlite:///$DB"
unset LAMINDB_INSTANCE_DB CKPT_PATH WANDB_NAME
export NUMBA_CACHE_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/numba TRITON_CACHE_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/triton TORCH_EXTENSIONS_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/torch_extensions XDG_CACHE_HOME=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/xdg MPLCONFIGDIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/matplotlib TMPDIR=$TMPBASE
mkdir -p "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR/wandb"
TCLSH=/lustre/fshomisc/sup/spack_soft/tcl/8.6.12/gcc-11.3.1-q2dfxcqelx36bbjbos7m22mn62tp5z2g/bin/tclsh
MODULECMD=/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup/libexec/modulecmd.tcl
export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles
eval "$("$TCLSH" "$MODULECMD" bash load cuda/12.8.0)"
WANDB_RUN_ID=$("$ENV/bin/python" -c 'import wandb; print(wandb.util.generate_id())')
export WANDB_RUN_ID WANDB_MODE=online WANDB_RESUME=never WANDB_DIR=$TMPBASE/wandb
if test "$SEGMENT" -eq 0; then timeout 300 "$ENV/bin/python" -c 'import wandb; api=wandb.Api(timeout=60); run=api.run("ml4ig/scprint_v2/r7atjv1s"); assert run.id == "r7atjv1s"; print("WANDB_CONNECTIVITY_PASS historical_run=" + run.id)'; fi
BEGIN=$("$ENV/bin/python" "$CONTROL/manage_primary_lineage.py" begin "$RUNROOT" "$ACTIVE" "$SLURM_JOB_ID" "$SEGMENT" "$GROUP" "$WANDB_RUN_ID")
CHECKPOINT=$("$ENV/bin/python" -c 'import json,sys; print(json.loads(sys.argv[1])["checkpoint"] or "")' "$BEGIN")
SEGMENT_OPEN=1
"$ENV/bin/python" "$CONTROL/wandb_online_monitor.py" "$WANDB_RUN_ID" "$E/primary_long_wandb_hourly.jsonl" "$E/primary_long_wandb_segment_${SEGMENT}.json" --entity ml4ig --project scprint_v2 --interval-seconds 3600 --retry-seconds 60 >"$E/primary_long_wandb_segment_${SEGMENT}.log" 2>&1 & WANDB_MONITOR_PID=$!
LOGGER="[{\"class_path\":\"lightning.pytorch.loggers.WandbLogger\",\"init_args\":{\"project\":\"scprint_v2\",\"entity\":\"ml4ig\",\"save_dir\":\"$WANDB_DIR\",\"offline\":false,\"name\":\"primary-esi-segment-${SEGMENT}-${WANDB_RUN_ID}\",\"group\":\"$GROUP\"}}]"
CALLBACKS="[{\"class_path\":\"lightning.pytorch.callbacks.ModelCheckpoint\",\"init_args\":{\"monitor\":\"val_loss\",\"save_top_k\":5,\"save_last\":true}},{\"class_path\":\"stable_attention_guard.StableAttentionGuard\",\"init_args\":{\"receipt_path\":\"$E/primary_long_attention_segment_${SEGMENT}.json\"}},{\"class_path\":\"gpu_telemetry_guard.GPUTelemetryGuard\",\"init_args\":{\"receipt_path\":\"$E/primary_long_gpu_segment_${SEGMENT}.jsonl\",\"every_n_steps\":100}}"
if test "$SEGMENT" -eq 0; then CALLBACKS+=",{\"class_path\":\"initial_long_health_guard.InitialLongHealthGuard\",\"init_args\":{\"receipt_path\":\"$E/primary_long_initial_health.json\",\"probe_steps\":300}},{\"class_path\":\"generation_intermediate_guard.GenerationIntermediateGuard\",\"init_args\":{\"receipt_path\":\"$E/primary_long_generation.json\",\"batch_trace_path\":\"$E/primary_long_batch_order.jsonl\",\"instrument_from_step\":0,\"instrument_until_step\":300,\"stop_after_steps\":300,\"stop_training\":false}},{\"class_path\":\"throughput_guard.ThroughputGuard\",\"init_args\":{\"receipt_path\":\"$E/primary_long_throughput.json\",\"expected_steps\":200,\"max_seconds_per_step\":0.7776,\"batch_size\":50,\"world_size\":2}}"; fi
CALLBACKS+="]"
CMD=("$ENV/bin/scprint2" fit --config "$P/config/base_v3_ntv3_primary.yml" --seed_everything 42 --data.store_location "$CACHE" --data.gene_embeddings "$NTV3" --data.weight_scaler 500 --data.batch_size 50 --data.n_samples_per_epoch 1000000 --model.gene_pos_file "$GENE_LOCS" --trainer.precision bf16-mixed --trainer.logger "$LOGGER" --trainer.callbacks "$CALLBACKS" --trainer.log_every_n_steps 100 --trainer.num_sanity_val_steps 0 --trainer.default_root_dir "$RUNROOT")
if test -n "$CHECKPOINT"; then CMD+=(--ckpt_path "$CHECKPOINT"); fi
printf -v SERIALIZED '%q ' "${CMD[@]}"; if test "$SEGMENT" -eq 0; then [[ "$SERIALIZED" != *--ckpt_path* ]]; else [[ "$SERIALIZED" == *"hpc_ckpt_${SEGMENT}.ckpt"* ]]; fi
printf 'PRIMARY_LONG_START %s job=%s segment=%s commit=%s scd=%s random_init=%s wandb_id=%s group=%s checkpoint=%q wandb_ipc=%q\n' "$(date -Is)" "$SLURM_JOB_ID" "$SEGMENT" "$EXPECTED_SCPRINT_COMMIT" "$EXPECTED_SCDATALOADER_COMMIT" "$([[ $SEGMENT -eq 0 ]] && echo true || echo false)" "$WANDB_RUN_ID" "$GROUP" "${CHECKPOINT:-null}" "$WANDB_DIR"
set +e
srun --label --kill-on-bad-exit=1 "${CMD[@]}"
RC=$?
set -e
NEXT=$RUNROOT/hpc_ckpt_$((SEGMENT+1)).ckpt
if test "$RC" -eq 0 && test -s "$NEXT"; then EXIT_REASON=native-requeue-checkpoint-$((SEGMENT+1)); exit 0; fi
EXIT_REASON=training-exit-$RC
exit "$RC"
