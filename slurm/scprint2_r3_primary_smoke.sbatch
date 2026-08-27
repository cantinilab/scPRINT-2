#!/bin/bash
#SBATCH --job-name=scprint2-r3-primary-smoke
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-primary-smoke-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-primary-smoke-%j.err
#SBATCH --open-mode=append
#SBATCH --account=wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=22
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
set -euo pipefail
R=/lustre/fswork/projects/rech/xeg/uat95fg
P=$R/scPRINT-esi-primary-t_6cff2510
S=$R/scDataLoader-esi-primary-t_6cff2510
ENV=$R/scPRINT/.venv
E=$R/scprint2_basev3_revision3_t_6cff2510
CONTROL=$P/scripts/training
DB=$R/scprint2_base_v3_filt_esi_t_6cff2510.lndb
CACHE=/lustre/fsn1/projects/rech/xeg/uat95fg/base_v3_filt_esi_t_6cff2510
NTV3=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_inputs/t_2bbe0ae6/ntv3-full/gene_embeddings.parquet
GENE_LOCS=$R/scPRINT/data/main/gene_locs/all_locs.parquet
CACHE_RECEIPT=$E/primary_filtered_cache_manifest.json
EXPECTED_SCPRINT_COMMIT=${EXPECTED_SCPRINT_COMMIT:?submitter must bind the reviewed PR head}
SCD_COMMIT=0a2fb2080dd2bf471c1532c04ee0789ce06441c2
ROOT=$E/primary_smoke_${SLURM_JOB_ID}
THROUGHPUT_ROOT=$E/primary_smoke_throughput_${SLURM_JOB_ID}
TMPBASE=${JOBSCRATCH:-/tmp}/${SLURM_JOB_ID}-wandb
cleanup(){ rc=$?; rm -rf "$TMPBASE"; printf 'PRIMARY_SMOKE_EXIT %s job=%s rc=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$rc"; }
trap cleanup EXIT
for path in "$P" "$S" "$DB" "$CACHE_RECEIPT" "$CONTROL/adopt_primary_filtered_cache.py" "$NTV3" "$GENE_LOCS"; do test -e "$path"; done
test "$(git -C "$P" rev-parse HEAD)" = "$EXPECTED_SCPRINT_COMMIT"
test "$(git -C "$S" rev-parse HEAD)" = "$SCD_COMMIT"
test -z "$(git -C "$P" status --porcelain)"; test -z "$(git -C "$S" status --porcelain)"
test ! -e "$ROOT"; test ! -e "$THROUGHPUT_ROOT"
"$ENV/bin/python" - "$CACHE_RECEIPT" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
assert receipt['status']=='accepted'
assert receipt['adoption']=='nine-existing-files-plus-derived-train-weights-no-rebuild'
assert receipt['actual']==receipt['expected']
assert len(receipt['cache_files'])==10
assert receipt['train_weights']['cardinality']==receipt['expected']['sampler_classes']
assert receipt['ntv3_sha256']=='762b474c37b6a911395cb9b87d3795ce876e417a9f5f65a7afe867bca52e964d'
print('PRIMARY_CACHE_GATE_PASS',receipt['actual'])
PY
mkdir -p "$ROOT" "$THROUGHPUT_ROOT" "$TMPBASE/wandb"
export PYTHONPATH="$CONTROL:$P:$S:$R"
unset LAMINDB_INSTANCE_DB CKPT_PATH WANDB_NAME
export LAMIN_CURRENT_INSTANCE=jkobject/scprint2 LAMINDB_DJANGO_DATABASE_URL="sqlite:///$DB"
export NUMBA_CACHE_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/numba TRITON_CACHE_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/triton TORCH_EXTENSIONS_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/torch_extensions XDG_CACHE_HOME=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/xdg MPLCONFIGDIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/matplotlib TMPDIR=$TMPBASE
mkdir -p "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
TCLSH=/lustre/fshomisc/sup/spack_soft/tcl/8.6.12/gcc-11.3.1-q2dfxcqelx36bbjbos7m22mn62tp5z2g/bin/tclsh
MODULECMD=/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup/libexec/modulecmd.tcl
export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles
eval "$("$TCLSH" "$MODULECMD" bash load cuda/12.8.0)"
export WANDB_MODE=online WANDB_RESUME=never WANDB_RUN_ID=p${SLURM_JOB_ID} WANDB_DIR=$TMPBASE/wandb
LOGGER="[{\"class_path\":\"lightning.pytorch.loggers.WandbLogger\",\"init_args\":{\"project\":\"scprint_v2\",\"entity\":\"ml4ig\",\"save_dir\":\"$WANDB_DIR\",\"offline\":false,\"name\":\"r3-primary-smoke-${SLURM_JOB_ID}\",\"group\":\"r3-primary-t_6cff2510\"}}]"
CALLBACKS="[{\"class_path\":\"lightning.pytorch.callbacks.ModelCheckpoint\",\"init_args\":{\"save_top_k\":0,\"save_last\":false}},{\"class_path\":\"finite_training_guard.FiniteTrainingGuard\",\"init_args\":{\"receipt_path\":\"$ROOT/finite.json\",\"required_train_steps\":300}},{\"class_path\":\"stable_attention_guard.StableAttentionGuard\",\"init_args\":{\"receipt_path\":\"$ROOT/stable_attention.json\"}},{\"class_path\":\"generation_intermediate_guard.GenerationIntermediateGuard\",\"init_args\":{\"receipt_path\":\"$ROOT/generation.json\",\"batch_trace_path\":\"$ROOT/batch_order.jsonl\",\"instrument_from_step\":0,\"stop_after_steps\":300}},{\"class_path\":\"gpu_telemetry_guard.GPUTelemetryGuard\",\"init_args\":{\"receipt_path\":\"$ROOT/gpu.jsonl\",\"every_n_steps\":100}}]"
BASE=("$ENV/bin/scprint2" fit --config "$P/config/base_v3_ntv3_primary.yml" --seed_everything 42 --data.store_location "$CACHE" --data.gene_embeddings "$NTV3" --data.weight_scaler 500 --data.batch_size 50 --data.n_samples_per_epoch 1000000 --model.gene_pos_file "$GENE_LOCS" --trainer.precision bf16-mixed --trainer.num_sanity_val_steps 0)
CMD=("${BASE[@]}" --trainer.logger "$LOGGER" --trainer.callbacks "$CALLBACKS" --trainer.limit_train_batches 300 --trainer.max_steps 300 --trainer.limit_val_batches 10 --trainer.log_every_n_steps 10 --trainer.default_root_dir "$ROOT")
printf -v SERIALIZED '%q ' "${CMD[@]}"; [[ "$SERIALIZED" != *--ckpt_path* ]]; [[ "$SERIALIZED" == *base_v3_ntv3_primary.yml* ]]
printf 'PRIMARY_SMOKE_START %s job=%s commit=%s scd=%s random_init=true ranks=2 guarded_steps=300 validation=10 wandb_ipc=%q\n' "$(date -Is)" "$SLURM_JOB_ID" "$EXPECTED_SCPRINT_COMMIT" "$SCD_COMMIT" "$WANDB_DIR"
srun --label --kill-on-bad-exit=1 "${CMD[@]}"
"$ENV/bin/python" - "$ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for path in (root/'finite.json',root/'finite.rank_1.json'):
 receipt=json.load(open(path)); assert receipt['train_batches']>=300 and receipt['gradient_checks']>=300 and receipt['parameter_checks']>=300 and receipt['validation_batches']>=10 and receipt['validation_embeddings_finite'] and receipt['coordinated_skipped_batches']==0
for rank in (0,1):
 attention=json.load(open(root/f'stable_attention.rank_{rank}.json')); assert attention['status']=='accepted_native_sdpa' and not attention['legacy_triton_modules']
rows=[json.loads(line) for line in (root/'gpu.jsonl').read_text().splitlines()]
assert len(rows)>=4
assert all(row['metrics']['fallback/gpu_count']==2 for row in rows)
print('PRIMARY_SMOKE_GUARDS_PASS telemetry_rows='+str(len(rows)))
PY
"$ENV/bin/python" "$CONTROL/verify_online_smoke_wandb.py" "$WANDB_RUN_ID" "$ROOT/gpu.jsonl" "$ROOT/wandb_online_receipt.json" --entity ml4ig --project scprint_v2 --required-step 300
unset WANDB_RUN_ID WANDB_DIR
TP_CALLBACKS="[{\"class_path\":\"throughput_guard.ThroughputGuard\",\"init_args\":{\"receipt_path\":\"$THROUGHPUT_ROOT/throughput.json\",\"expected_steps\":200,\"max_seconds_per_step\":0.7776,\"batch_size\":50,\"world_size\":2}}]"
TP_CMD=("${BASE[@]}" --trainer.logger false --trainer.callbacks "$TP_CALLBACKS" --trainer.limit_train_batches 200 --trainer.max_steps 200 --trainer.limit_val_batches 0 --trainer.log_every_n_steps 200 --trainer.default_root_dir "$THROUGHPUT_ROOT")
srun --label --kill-on-bad-exit=1 "${TP_CMD[@]}"
"$ENV/bin/python" - "$THROUGHPUT_ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for rank in (0,1):
 receipt=json.load(open(root/f'throughput.rank_{rank}.json')); assert receipt['accepted'] and receipt['steps']==200 and receipt['world_size']==2,receipt
print('PRIMARY_SMOKE_THROUGHPUT_PASS')
PY
printf 'PRIMARY_SMOKE_PASS %s job=%s guarded_steps=300 validation_batches=10 throughput_steps=200\n' "$(date -Is)" "$SLURM_JOB_ID"
