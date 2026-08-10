#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <main|transcriptformer> <clean-notebook> <notebook-path>" >&2
  exit 2
fi

ENV_KIND="$1"
CLEAN_NOTEBOOK="$2"
NOTEBOOK="$3"

if [[ ! -f "${CLEAN_NOTEBOOK}" ]]; then
  echo "missing clean notebook: ${CLEAN_NOTEBOOK}" >&2
  exit 1
fi

cp -- "${CLEAN_NOTEBOOK}" "${NOTEBOOK}"
exec slurm/run_op_scib_notebook.sh "${ENV_KIND}" "${NOTEBOOK}"
