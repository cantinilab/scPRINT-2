#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=slurm/task3-paper-scib-%j.out
#SBATCH --error=slurm/task3-paper-scib-%j.out

set -euo pipefail
set +u
source /etc/profile
set -u
module load r/4.4.1

METHOD="$1"
EMBEDDING="$2"
EMBEDDING_KEY="$3"
WORK_ROOT="${WORK:-/lustre/fswork/projects/rech/xeg/${USER}}"
REPO_ROOT="${SCPRINT_REPO:-${WORK_ROOT}/scPRINT}"
SOURCE="${TASK3_INPUT:-${REPO_ROOT}/notebooks/scPRINT-2-repro-notebooks/data/task_3_embed.h5ad}"
LABEL_KEY="${TASK3_LABEL_KEY:-cell_type_ontology_term_id}"
PYTHON="${PAPER_SCIB_ENV:-${WORK_ROOT}/venvs/task3-paper-scib-1.1.3}/bin/python"
R_LIBS_USER="${PAPER_SCIB_R_LIB:-${WORK_ROOT}/R/task3-paper-scib-1.1.3}:${WORK_ROOT}/R/library/4.4"
export R_LIBS_USER
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RESULT_ROOT="${REPO_ROOT}/data/results/cross_species_embedding/paper_scib_1.1.3/${METHOD}"
SCRIPT_ROOT="${RESULT_ROOT}/script"
if [[ -e "${RESULT_ROOT}/COMPLETE" ]]; then
  echo "Refusing to overwrite completed paper-scIB result: ${RESULT_ROOT}" >&2
  exit 3
fi
mkdir -p "$RESULT_ROOT"

ARCHIVE="${RESULT_ROOT}/figshare_50760384.tar.gz"
if [[ ! -s "$ARCHIVE" ]]; then
  curl --fail --location --retry 3 \
    https://ndownloader.figshare.com/files/50760384 \
    --output "$ARCHIVE"
fi
if [[ ! -d "$SCRIPT_ROOT" ]]; then
  mkdir -p "$SCRIPT_ROOT"
  tar --extract --gzip --file "$ARCHIVE" --directory "$SCRIPT_ROOT" \
    --strip-components=1 --exclude='*/.git' --exclude='*/.git/*'
fi

# Figshare file 50760384 currently contains a top-level
# benchmark_project_script directory after the archive prefix is removed.
# Keep accepting a flat extraction as well so an already-staged archive can
# be resumed without changing the evaluator sources.
EVALUATOR_ROOT="$SCRIPT_ROOT"
if [[ -d "${SCRIPT_ROOT}/benchmark_project_script" ]]; then
  EVALUATOR_ROOT="${SCRIPT_ROOT}/benchmark_project_script"
fi
# The archived Python entry points intentionally use paths relative to both
# the entry-point directory and core_script/. With the archive's extra wrapper
# directory, expose the prepared paper data/output at the expected parent.
for directory in data output; do
  if [[ ! -e "${SCRIPT_ROOT}/${directory}" ]]; then
    ln -s "../${directory}" "${SCRIPT_ROOT}/${directory}"
  fi
done

# The deposited scib_metric_running.py contains a stray non-Python "ß" after
# the task19 Planaria branch. It prevents parsing even for task3. Remove only
# that unreachable syntax typo, then prove the deposited entry point compiles.
SCIB_ENTRYPOINT="${EVALUATOR_ROOT}/scib_metric_running.py"
sed -i "s/]  ß$/]/" "$SCIB_ENTRYPOINT"
"$PYTHON" -m py_compile "$SCIB_ENTRYPOINT"

"$PYTHON" "$REPO_ROOT/scripts/prepare_task3_paper_scib.py" \
  --source "$SOURCE" \
  --embedding "$EMBEDDING" \
  --embedding-key "$EMBEDDING_KEY" \
  --paper-root "$RESULT_ROOT" \
  --batch-key orig.ident \
  --label-key "$LABEL_KEY"

cd "$EVALUATOR_ROOT"
EVALUATION_ROOT="${RESULT_ROOT}/output/evaluation/saturn"
if [[ ! -s "${EVALUATION_ROOT}/task3_saturn_ARI.txt" || \
      ! -s "${EVALUATION_ROOT}/task3_saturn_lisi_batch_40.txt" || \
      ! -s "${EVALUATION_ROOT}/task3_saturn_lisi_celltype_40.txt" ]]; then
  Rscript "$REPO_ROOT/scripts/run_task3_paper_ari_lisi.R" "$RESULT_ROOT"
fi
if [[ ! -s "${EVALUATION_ROOT}/task3_saturn_ASW_metric.csv" || \
      ! -s "${EVALUATION_ROOT}/task3_saturn_NMI.csv" ]]; then
  "$PYTHON" emb_recon_ASW_NMI.py --all_targets task3 --methods saturn
fi
if [[ ! -s "${EVALUATION_ROOT}/task3_saturn_scib_output.csv" ]]; then
  "$PYTHON" scib_metric_running.py --method saturn --target task3
fi
"$PYTHON" "$REPO_ROOT/scripts/collect_task3_paper_scib.py" \
  --paper-root "$RESULT_ROOT" \
  --method "$METHOD" \
  --reference "$REPO_ROOT/data/reference/cross_species_task3_paper_scores.csv"
touch "$RESULT_ROOT/COMPLETE"
