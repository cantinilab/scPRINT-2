#!/bin/bash
#SBATCH --job-name=scprint2-r3-runtime
#SBATCH --partition=compil
#SBATCH --account=wbg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-runtime-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scprint2-r3-runtime-%j.err
set -euo pipefail
R=/lustre/fswork/projects/rech/xeg/uat95fg
P=$R/scPRINT-esi-primary-t_6cff2510
S=$R/scDataLoader-esi-primary-t_6cff2510
UV_BIN=$HOME/.local/bin/uv
EXPECTED_SCPRINT_COMMIT=${EXPECTED_SCPRINT_COMMIT:?submitter must bind reviewed PR head}
EXPECTED_SCDATALOADER_COMMIT=0a2fb2080dd2bf471c1532c04ee0789ce06441c2
RUNTIME_ROOT=${RUNTIME_ROOT:?submitter must bind a new absent runtime root}
test -x "$UV_BIN"
test ! -e "$RUNTIME_ROOT"
test "$(git -C "$P" rev-parse HEAD)" = "$EXPECTED_SCPRINT_COMMIT"
test "$(git -C "$S" rev-parse HEAD)" = "$EXPECTED_SCDATALOADER_COMMIT"
test -z "$(git -C "$P" status --porcelain)"
test -z "$(git -C "$S" status --porcelain)"
mkdir -p "$RUNTIME_ROOT" /lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/uv
export UV_PROJECT_ENVIRONMENT=$RUNTIME_ROOT/.venv
export UV_CACHE_DIR=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint2_runtime_cache/uv
export UV_PYTHON_INSTALL_DIR=$RUNTIME_ROOT/python
export UV_PYTHON_PREFERENCE=only-managed
printf 'FROZEN_RUNTIME_BUILD_START %s commit=%s scd=%s uv=%s root=%s\n' "$(date -Is)" "$EXPECTED_SCPRINT_COMMIT" "$EXPECTED_SCDATALOADER_COMMIT" "$($UV_BIN --version)" "$RUNTIME_ROOT"
"$UV_BIN" python install 3.12
"$UV_BIN" sync --frozen --no-dev --project "$P" --python 3.12
python3 "$P/scripts/training/verify_frozen_runtime.py" create "$RUNTIME_ROOT" "$P" "$S" "$EXPECTED_SCPRINT_COMMIT" "$EXPECTED_SCDATALOADER_COMMIT" --uv-bin "$UV_BIN"
python3 "$P/scripts/training/verify_frozen_runtime.py" verify "$RUNTIME_ROOT" "$P" "$S" "$EXPECTED_SCPRINT_COMMIT" "$EXPECTED_SCDATALOADER_COMMIT"
printf 'FROZEN_RUNTIME_BUILD_PASS %s root=%s\n' "$(date -Is)" "$RUNTIME_ROOT"
