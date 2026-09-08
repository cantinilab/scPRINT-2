#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(6, 7))) {
  stop(paste(
    "Usage: convert_pandora_lung_rds.R INPUT.rds GENE_MAPPING.tsv OUTPUT.h5ad",
    "ORGANISM_ONTOLOGY_ID SPECIES PYTHON [TARGET_FEATURE_SPACE]"
  ))
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
ortholog_path <- normalizePath(args[[2]], mustWork = TRUE)
output_path <- args[[3]]
organism_id <- args[[4]]
species <- args[[5]]
python <- args[[6]]
target_feature_space <- if (length(args) == 7) args[[7]] else "human_one2one"
if (!file.exists(python)) stop("Python executable does not exist: ", python)

object <- readRDS(input_path)
if (!"Seurat" %in% class(object)) stop("Expected a Seurat RDS object")
metadata <- tryCatch(
  slot(object, "meta.data"),
  error = function(e) stop("Cannot read Seurat meta.data slot: ", conditionMessage(e))
)
if (!"celltype" %in% colnames(metadata)) {
  stop("Pandora RDS has no celltype annotation")
}
if (any(is.na(metadata$celltype) | metadata$celltype == "")) {
  stop("Pandora celltype annotation contains missing values")
}

assays <- tryCatch(
  slot(object, "assays"),
  error = function(e) stop("Cannot read Seurat assays slot: ", conditionMessage(e))
)
if (!"RNA" %in% names(assays)) stop("Pandora RDS has no RNA assay")
rna <- assays[["RNA"]]
counts <- tryCatch(
  slot(rna, "counts"),
  error = function(e) stop("Cannot read RNA counts slot: ", conditionMessage(e))
)
if (!"dgCMatrix" %in% class(counts)) stop("Pandora RNA counts are not a dgCMatrix")
dimensions <- slot(counts, "Dim")
if (dimensions[[2]] != nrow(metadata)) stop("RNA count columns do not match metadata rows")
dimnames <- slot(counts, "Dimnames")
if (length(dimnames) != 2 || is.null(dimnames[[1]])) stop("Counts lack gene names")
rownames(metadata) <- make.unique(paste(species, rownames(metadata), sep = "_"))

tmp <- tempfile("pandora-h5ad-")
dir.create(tmp)
on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
data_path <- file.path(tmp, "data.f64")
indices_path <- file.path(tmp, "indices.i32")
indptr_path <- file.path(tmp, "indptr.i32")
genes_path <- file.path(tmp, "genes.txt")
obs_path <- file.path(tmp, "obs.tsv")
writeBin(as.double(slot(counts, "x")), data_path, size = 8, endian = "little")
writeBin(as.integer(slot(counts, "i")), indices_path, size = 4, endian = "little")
writeBin(as.integer(slot(counts, "p")), indptr_path, size = 4, endian = "little")
writeLines(dimnames[[1]], genes_path, useBytes = TRUE)
write.table(metadata, obs_path, sep = "\t", quote = FALSE, col.names = NA)

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(file_arg) != 1) stop("Cannot resolve this script's location")
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
script <- file.path(script_dir, "prepare_pandora_lung_atlas.py")
command <- c(
  script, "assemble-csc",
  "--data", data_path,
  "--indices", indices_path,
  "--indptr", indptr_path,
  "--genes", genes_path,
  "--obs", obs_path,
  "--orthologs", ortholog_path,
  "--output", output_path,
  "--organism-id", organism_id,
  "--species", species,
  "--target-feature-space", target_feature_space
)
status <- system2(python, command)
if (status != 0) stop("Python h5ad assembly failed with status ", status)
message("Converted ", basename(input_path), ": ", dimensions[[2]], " cells")
