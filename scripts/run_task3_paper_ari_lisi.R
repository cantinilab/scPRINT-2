args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: Rscript run_task3_paper_ari_lisi.R <paper-result-root>")
}

root <- normalizePath(args[[1]], mustWork = TRUE)
script_root <- file.path(root, "script")
if (dir.exists(file.path(script_root, "benchmark_project_script"))) {
  script_root <- file.path(script_root, "benchmark_project_script")
}
source(file.path(script_root, "core_script", "evaluation_ARI_LISI.R"))
run_all_metric(
  "task3",
  "saturn",
  pre_dir = file.path(root, "output", "method_outputs", "saturn", ""),
  save_dir = file.path(root, "output", "evaluation", "saturn", "")
)
