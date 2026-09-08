from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from scprint2.preprocessing.pandora import (
    assemble_csc_h5ad,
    assemble_h5ad,
    canonical_mouse_symbol,
    combine_h5ads,
    filter_h5ad_by_obs_labels,
    summarize_pandora_annotations,
    summarize_pandora_orthology,
)


def test_canonical_mouse_symbol_removes_only_make_unique_suffixes():
    assert canonical_mouse_symbol("Ptgr1.4") == "Ptgr1"
    assert canonical_mouse_symbol("H2-D1") == "H2-D1"
    assert canonical_mouse_symbol("symbol.alpha") == "symbol.alpha"


def test_pandora_notebook_summaries_keep_annotation_and_orthology_counts():
    annotations = pd.DataFrame(
        {
            "species": ["Cat", "Cat", "Tiger"],
            "cell_type": ["AT1", "AT2", "AT1"],
            "cells": [10, 5, 7],
        }
    )
    orthologs = pd.DataFrame(
        {
            "mouse_gene_symbol": ["A", "B"],
            "mouse_ensembl_gene_id": ["ENSMUSG1", "ENSMUSG2"],
            "human_ensembl_gene_id": ["ENSG1", "ENSG2"],
            "human_homology_type": ["ortholog_one2one", "ortholog_one2one"],
            "human_orthology_confidence": ["1", "1"],
        }
    )

    annotation_summary = summarize_pandora_annotations(annotations)
    orthology_summary = summarize_pandora_orthology(orthologs)

    assert annotation_summary.set_index("species").loc["Cat", "cells"] == 15
    assert annotation_summary.set_index("species").loc["Cat", "cell_types"] == 2
    assert orthology_summary["one_to_one_rows"] == 2
    assert orthology_summary["duplicate_mouse_symbols"] == 0


def test_assemble_h5ad_adds_required_scprint_metadata(tmp_path: Path):
    matrix_path = tmp_path / "counts.mtx"
    obs_path = tmp_path / "obs.tsv"
    var_path = tmp_path / "var.tsv"
    output_path = tmp_path / "result.h5ad"
    scipy.io.mmwrite(matrix_path, sp.csr_matrix([[1, 0], [2, 3]]))
    pd.DataFrame(
        {"celltype": ["AT1", "Unknown"]}, index=["Rabbit_a", "Rabbit_b"]
    ).to_csv(obs_path, sep="\t")
    pd.DataFrame(
        {"ensembl_gene_id": ["ENSG1", "ENSG2"]}, index=["ENSG1", "ENSG2"]
    ).to_csv(var_path, sep="\t")

    result = assemble_h5ad(
        matrix_path,
        obs_path,
        var_path,
        output_path,
        "NCBITaxon:9986",
        "Rabbit",
        source_count_sum=8,
        source_feature_count=3,
        mapped_source_feature_count=2,
    )

    assert result.shape == (2, 2)
    assert result.X.toarray().tolist() == [[1, 2], [0, 3]]
    assert result.obs["cell_type_ontology_term_id"].tolist() == [
        "CL:0002062",
        "unknown",
    ]
    assert result.obs["has_benchmark_label"].tolist() == [True, False]
    assert set(result.obs["source_organism_ontology_term_id"]) == {
        "NCBITaxon:9986"
    }
    assert set(result.obs["organism_ontology_term_id"]) == {"NCBITaxon:9606"}
    assert result.var_names.tolist() == ["ENSG1", "ENSG2"]
    assert set(result.var["organism"]) == {"NCBITaxon:9606"}
    assert result.uns["pandora_gene_mapping"]["target_feature_organism"] == (
        "NCBITaxon:9606"
    )
    assert result.uns["pandora_gene_mapping"]["mapped_count_sum"] == 6
    assert result.uns["pandora_gene_mapping"]["count_retention_fraction"] == 0.75
    assert output_path.exists()


def test_assemble_csc_maps_and_sums_duplicate_human_targets(tmp_path: Path):
    counts = sp.csc_matrix(np.array([[1, 0], [2, 3], [4, 5]], dtype=np.float64))
    data_path = tmp_path / "data.f64"
    indices_path = tmp_path / "indices.i32"
    indptr_path = tmp_path / "indptr.i32"
    counts.data.astype("<f8").tofile(data_path)
    counts.indices.astype("<i4").tofile(indices_path)
    counts.indptr.astype("<i4").tofile(indptr_path)
    genes_path = tmp_path / "genes.txt"
    genes_path.write_text("GeneA\nGeneA.1\nGeneB\n")
    obs_path = tmp_path / "obs.tsv"
    pd.DataFrame(
        {"celltype": ["AT1", "AT2"]}, index=["Rabbit_1", "Rabbit_2"]
    ).to_csv(obs_path, sep="\t")
    ortholog_path = tmp_path / "orthologs.tsv"
    pd.DataFrame(
        {
            "mouse_gene_symbol": ["GeneA", "GeneB"],
            "human_ensembl_gene_id": ["ENSG000001", "ENSG000002"],
        }
    ).to_csv(ortholog_path, sep="\t", index=False)

    result = assemble_csc_h5ad(
        data_path,
        indices_path,
        indptr_path,
        genes_path,
        obs_path,
        ortholog_path,
        tmp_path / "mapped.h5ad",
        "NCBITaxon:9986",
        "Rabbit",
    )

    np.testing.assert_array_equal(
        result.X.toarray(), np.array([[3, 4], [3, 5]], dtype=np.float32)
    )
    assert result.var_names.tolist() == ["ENSG000001", "ENSG000002"]
    assert result.uns["pandora_gene_mapping"]["mapped_count_sum"] == 15


def test_assemble_csc_mouse_mode_maps_symbols_to_mouse_ensembl(tmp_path: Path):
    counts = sp.csc_matrix(
        np.array(
            [
                [1, 0],
                [2, 3],
                [4, 5],
                [7, 11],
            ],
            dtype=np.float64,
        )
    )
    data_path = tmp_path / "data.f64"
    indices_path = tmp_path / "indices.i32"
    indptr_path = tmp_path / "indptr.i32"
    counts.data.astype("<f8").tofile(data_path)
    counts.indices.astype("<i4").tofile(indices_path)
    counts.indptr.astype("<i4").tofile(indptr_path)
    genes_path = tmp_path / "genes.txt"
    genes_path.write_text("GeneA\nGeneA.1\nGeneB\nUnmapped\n")
    obs_path = tmp_path / "obs.tsv"
    pd.DataFrame(
        {"celltype": ["Dendritic cells", "Immune cells"]},
        index=["Rabbit_1", "Rabbit_2"],
    ).to_csv(obs_path, sep="\t")
    mapping_path = tmp_path / "mouse_mapping.tsv"
    pd.DataFrame(
        {
            "mouse_gene_symbol": ["GeneA", "GeneB"],
            "mouse_ensembl_gene_id": ["ENSMUSG000001", "ENSMUSG000002"],
            "human_ensembl_gene_id": ["ENSG000001", "ENSG000002"],
        }
    ).to_csv(mapping_path, sep="\t", index=False)

    result = assemble_csc_h5ad(
        data_path,
        indices_path,
        indptr_path,
        genes_path,
        obs_path,
        mapping_path,
        tmp_path / "mouse.h5ad",
        "NCBITaxon:9986",
        "Rabbit",
        target_feature_space="mouse_homolog",
    )

    np.testing.assert_array_equal(
        result.X.toarray(), np.array([[3, 4], [3, 5]], dtype=np.float32)
    )
    assert result.var_names.tolist() == ["ENSMUSG000001", "ENSMUSG000002"]
    assert set(result.var["organism"]) == {"NCBITaxon:10090"}
    assert set(result.obs["source_organism_ontology_term_id"]) == {"NCBITaxon:9986"}
    assert set(result.obs["organism_ontology_term_id"]) == {"NCBITaxon:10090"}
    assert result.obs["species"].tolist() == ["Rabbit", "Rabbit"]
    assert result.obs["cell_type_ontology_term_id"].tolist() == [
        "CL:0000451",
        "CL:0000738",
    ]
    assert result.uns["pandora_gene_mapping"]["target_feature_organism"] == (
        "NCBITaxon:10090"
    )
    assert result.uns["pandora_gene_mapping"]["mapping_column"] == (
        "mouse_ensembl_gene_id"
    )
    assert result.uns["pandora_gene_mapping"]["source_count_sum"] == 33
    assert result.uns["pandora_gene_mapping"]["mapped_source_feature_count"] == 3
    assert result.uns["pandora_gene_mapping"]["mapped_count_sum"] == 15


def test_unknown_and_mix_cell_types_remain_unknown(tmp_path: Path):
    matrix_path = tmp_path / "counts.mtx"
    obs_path = tmp_path / "obs.tsv"
    var_path = tmp_path / "var.tsv"
    output_path = tmp_path / "unknowns.h5ad"
    scipy.io.mmwrite(matrix_path, sp.csr_matrix([[1, 2]]))
    pd.DataFrame({"celltype": ["Unknown", "Mix"]}, index=["cell_a", "cell_b"]).to_csv(
        obs_path, sep="\t"
    )
    pd.DataFrame({"ensembl_gene_id": ["ENSG1"]}, index=["ENSG1"]).to_csv(
        var_path, sep="\t"
    )

    result = assemble_h5ad(
        matrix_path,
        obs_path,
        var_path,
        output_path,
        "NCBITaxon:9986",
        "Rabbit",
    )

    assert result.obs["cell_type_ontology_term_id"].tolist() == [
        "unknown",
        "unknown",
    ]
    assert result.obs["has_benchmark_label"].tolist() == [False, False]


def test_filter_h5ad_by_obs_labels_writes_physical_filtered_atlas(tmp_path: Path):
    input_path = tmp_path / "full.h5ad"
    output_path = tmp_path / "filtered.h5ad"
    obs = pd.DataFrame(
        {
            "cell_type_ontology_term_id": [
                "CL:0002062",
                "unknown",
                " UnMapped ",
                "Mix",
                "CL:0002063",
            ],
            "species": ["cat", "cat", "dog", "dog", "cat"],
        },
        index=[f"cell-{i}" for i in range(5)],
    )
    adata = sp.csr_matrix(np.ones((5, 3), dtype=np.float32))
    import anndata as ad

    ad.AnnData(X=adata, obs=obs).write_h5ad(input_path)

    result = filter_h5ad_by_obs_labels(
        input_path,
        output_path,
        excluded_labels=("unknown", "unmapped", "mix"),
        expected_cells=2,
    )

    assert result.n_obs == 2
    assert result.obs_names.tolist() == ["cell-0", "cell-4"]
    assert output_path.exists()
    assert result.uns["pandora_cell_filter"]["source_cells"] == 5
    assert result.uns["pandora_cell_filter"]["filtered_cells"] == 2
    assert result.uns["pandora_cell_filter"]["excluded_cells"] == 3
    assert result.uns["pandora_cell_filter"]["physical_filter"] is True


def test_combine_preserves_single_feature_organism(tmp_path: Path):
    first_matrix = tmp_path / "first.mtx"
    first_obs = tmp_path / "first_obs.tsv"
    first_var = tmp_path / "first_var.tsv"
    second_matrix = tmp_path / "second.mtx"
    second_obs = tmp_path / "second_obs.tsv"
    second_var = tmp_path / "second_var.tsv"
    scipy.io.mmwrite(first_matrix, sp.csr_matrix([[1, 2]]))
    scipy.io.mmwrite(second_matrix, sp.csr_matrix([[3, 4]]))
    pd.DataFrame({"celltype": ["AT1", "AT2"]}, index=["rabbit_a", "rabbit_b"]).to_csv(
        first_obs, sep="\t"
    )
    pd.DataFrame({"celltype": ["AT1", "AT2"]}, index=["rat_a", "rat_b"]).to_csv(
        second_obs, sep="\t"
    )
    pd.DataFrame(
        {"ensembl_gene_id": ["ENSMUSG000001"]}, index=["ENSMUSG000001"]
    ).to_csv(first_var, sep="\t")
    pd.DataFrame(
        {"ensembl_gene_id": ["ENSMUSG000001"]}, index=["ENSMUSG000001"]
    ).to_csv(second_var, sep="\t")
    first_output = tmp_path / "first.h5ad"
    second_output = tmp_path / "second.h5ad"

    assemble_h5ad(
        first_matrix,
        first_obs,
        first_var,
        first_output,
        "NCBITaxon:9986",
        "Rabbit",
        feature_organism_id="NCBITaxon:10090",
    )
    assemble_h5ad(
        second_matrix,
        second_obs,
        second_var,
        second_output,
        "NCBITaxon:10116",
        "Rat",
        feature_organism_id="NCBITaxon:10090",
    )

    combined = combine_h5ads([first_output, second_output], tmp_path / "combined.h5ad")

    assert set(combined.var["organism"]) == {"NCBITaxon:10090"}
    assert set(combined.obs["organism_ontology_term_id"]) == {"NCBITaxon:10090"}
    assert set(combined.obs["source_organism_ontology_term_id"]) == {
        "NCBITaxon:9986",
        "NCBITaxon:10116",
    }
