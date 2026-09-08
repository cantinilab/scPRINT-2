from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scprint2.preprocessing.eye import (
    EYE_CELL_TYPE_ONTOLOGY,
    align_eye_to_checkpoint_genes,
    align_eye_to_reference_genes,
    combine_eye_native_species,
    combine_eye_ortholog_species,
    combine_eye_species,
    combine_eye_target_species,
    direct_symbol_to_ensembl_mapping,
    expand_observed_gene_aliases,
    extract_eye_annotations,
    load_geo_count_subset,
    merge_preferred_target_mappings,
    normalize_geo_cell_id,
    resolve_orthology_conflicts,
    resolve_source_orthology_targets,
    strict_one_to_one_mapping,
    update_eye_annotation_contract,
    write_eye_manifest,
    write_eye_native_gene_audit,
    write_eye_orthology_audit,
)


def test_normalize_geo_cell_id_only_rewrites_terminal_seurat_suffix():
    assert normalize_geo_cell_id("sample_barcode.1") == "sample_barcode-1"
    assert normalize_geo_cell_id("sample.1_barcode") == "sample.1_barcode"


def test_extract_eye_annotations_preserves_native_organism_and_known_labels():
    source = ad.AnnData(
        X=sp.csr_matrix([[1], [2]]),
        obs=pd.DataFrame(
            {
                "orig.ident": ["donor-1", "donor-2"],
                "celltype": ["Beam A", "Pericyte"],
            },
            index=["human_AAAC.1", "human_TTTG.1"],
        ),
        var=pd.DataFrame(index=["GENE"]),
    )

    annotations = extract_eye_annotations(
        source,
        species="human",
        source_organism_ontology_term_id="NCBITaxon:9606",
    )

    assert annotations["geo_cell_id"].tolist() == [
        "human_AAAC-1",
        "human_TTTG-1",
    ]
    assert annotations["source_organism_ontology_term_id"].unique().tolist() == [
        "NCBITaxon:9606"
    ]
    assert annotations["organism_ontology_term_id"].unique().tolist() == [
        "NCBITaxon:9606"
    ]
    assert annotations["cell_type_ontology_term_id"].tolist() == [
        "CL:0000327",
        "CL:0000669",
    ]


def test_extract_eye_annotations_fails_the_annotation_gate():
    source = ad.AnnData(
        X=sp.csr_matrix([[1]]),
        obs=pd.DataFrame(
            {"orig.ident": ["sample"], "celltype": ["unreviewed label"]},
            index=["cell.1"],
        ),
        var=pd.DataFrame(index=["GENE"]),
    )

    with pytest.raises(ValueError, match="lack ontology mappings"):
        extract_eye_annotations(
            source,
            species="human",
            source_organism_ontology_term_id="NCBITaxon:9606",
        )


def test_update_eye_annotation_contract_uses_supported_macaque_proxy():
    annotations = _annotations(
        "macaque_fascicularis",
        ["macaque-a-1", "macaque-b-1"],
        organism="NCBITaxon:9541",
    )
    annotations["celltype"] = ["Beam A", "JCT"]

    result = update_eye_annotation_contract(
        annotations,
        species="macaque_fascicularis",
        source_organism_ontology_term_id="NCBITaxon:9541",
        feature_organism_ontology_term_id="NCBITaxon:9544",
    )

    assert result["source_organism_ontology_term_id"].unique().tolist() == [
        "NCBITaxon:9541"
    ]
    assert result["organism_ontology_term_id"].unique().tolist() == [
        "NCBITaxon:9544"
    ]
    assert result["cell_type_ontology_term_id"].tolist() == [
        "CL:0000327",
        "CL:0002320",
    ]


def test_all_eye_cell_type_terms_are_known_to_small_v2():
    import torch

    checkpoint = torch.load(
        Path(__file__).with_name("small-v2.ckpt"),
        map_location="cpu",
        weights_only=False,
    )
    decoder = checkpoint["hyper_parameters"]["label_decoders"][
        "cell_type_ontology_term_id"
    ]
    known = set(map(str, decoder.values()))

    assert set(EYE_CELL_TYPE_ONTOLOGY.values()) <= known


def test_strict_one_to_one_mapping_drops_source_and_target_conflicts():
    first = pd.DataFrame(
        {
            "human_gene": ["A", "B", "C", "D"],
            "mouse_gene": ["a", "b", "c", "d"],
        }
    )
    second = pd.DataFrame(
        {
            "human_gene": ["A", "B", "E", "F"],
            "mouse_gene": ["a", "x", "d", "f"],
        }
    )

    result, audit = strict_one_to_one_mapping(
        [first, second], source_col="human_gene", target_col="mouse_gene"
    )

    assert result.to_dict("records") == [
        {"source_gene": "A", "mouse_gene": "a"},
        {"source_gene": "C", "mouse_gene": "c"},
        {"source_gene": "F", "mouse_gene": "f"},
    ]
    assert audit == {
        "input_rows": 8,
        "unique_pairs": 7,
        "conflicting_source_genes": 1,
        "conflicting_mouse_genes": 1,
        "retained_pairs": 3,
    }


def test_resolve_orthology_conflicts_prefers_one2one_then_identity():
    table = pd.DataFrame(
        {
            "source": ["A", "B", "C", "D", "E"],
            "mouse": ["m1", "m1", "m2", "m3", "m3"],
            "homology_type": [
                "ortholog_one2many",
                "ortholog_one2one",
                "ortholog_one2one",
                "ortholog_many2many",
                "ortholog_many2many",
            ],
            "source_homolog_percent_identity": [99, 80, 90, 70, 75],
        }
    )

    result, audit = resolve_orthology_conflicts(
        [table], source_col="source", target_col="mouse"
    )

    assert result.to_dict("records") == [
        {"source_gene": "B", "mouse_gene": "m1"},
        {"source_gene": "C", "mouse_gene": "m2"},
        {"source_gene": "E", "mouse_gene": "m3"},
    ]
    assert audit["retained_one2one"] == 2
    assert audit["retained_non_one2one"] == 1


def test_direct_human_symbol_mapping_normalizes_and_aggregates_macaque_suffixes():
    registry = pd.DataFrame(
        {
            "symbol": ["AAK1", "AAK1", "RP11-34P13.3"],
            "ensembl": ["ENSG2", "ENSG1", "ENSG3"],
        }
    )

    mapping, audit = direct_symbol_to_ensembl_mapping(
        registry,
        observed_genes=["AAK1_n", "AAK1_p", "RP11-34P13.3", "MSTRG.1"],
        symbol_col="symbol",
        ensembl_col="ensembl",
        normalize_terminal_np_suffixes=True,
    )

    assert mapping.to_dict("records") == [
        {"source_gene": "AAK1_n", "target_gene": "ENSG1"},
        {"source_gene": "AAK1_p", "target_gene": "ENSG1"},
        {"source_gene": "RP11-34P13.3", "target_gene": "ENSG3"},
    ]
    assert audit["terminal_np_suffix_hits"] == 2
    assert audit["ambiguous_registry_symbols"] == 1
    assert audit["collapsed_target_genes"] == 1


def test_resolve_source_orthology_targets_allows_many_sources_per_human_ensg():
    table = pd.DataFrame(
        {
            "mouse": ["m1", "m1", "m2"],
            "human_ensg": ["E2", "E1", "E1"],
            "homology_type": [
                "ortholog_one2many",
                "ortholog_one2one",
                "ortholog_one2one",
            ],
            "source_homolog_percent_identity": [99, 80, 90],
        }
    )

    mapping, audit = resolve_source_orthology_targets(
        [table], source_col="mouse", target_col="human_ensg"
    )

    assert mapping.to_dict("records") == [
        {"source_gene": "m1", "target_gene": "E1"},
        {"source_gene": "m2", "target_gene": "E1"},
    ]
    assert audit["conflicting_source_genes"] == 1
    assert audit["conflicting_target_genes"] == 1


def test_merge_preferred_target_mappings_uses_orthology_only_for_unmapped_ids():
    preferred = pd.DataFrame(
        {"source_gene": ["TP53", "SHARED"], "target_gene": ["ENSG1", "ENSG2"]}
    )
    fallback = pd.DataFrame(
        {
            "source_gene": ["ENSSSCG1", "SHARED", "ENSSSCG2"],
            "target_gene": ["ENSG1", "ENSG9", "ENSG3"],
        }
    )

    mapping, audit = merge_preferred_target_mappings(preferred, fallback)

    assert mapping.to_dict("records") == [
        {"source_gene": "ENSSSCG1", "target_gene": "ENSG1"},
        {"source_gene": "TP53", "target_gene": "ENSG1"},
        {"source_gene": "SHARED", "target_gene": "ENSG2"},
        {"source_gene": "ENSSSCG2", "target_gene": "ENSG3"},
    ]
    assert audit["fallback_input_pairs"] == 3
    assert audit["fallback_added_pairs"] == 2
    assert audit["retained_targets"] == 3


def test_expand_observed_gene_aliases_uses_symbols_and_ensembl_ids():
    mapping = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "ensembl": ["ENS1", "ENS2"],
            "target": ["T1", "T2"],
        }
    )

    result = expand_observed_gene_aliases(
        [mapping],
        source_columns=["symbol", "ensembl"],
        observed_genes=["A", "ENS2", "unmapped"],
    )

    assert [frame["_observed_source_gene"].tolist() for frame in result] == [
        ["A"],
        ["ENS2"],
    ]


def _annotations(
    species: str,
    cell_ids: list[str],
    organism: str = "NCBITaxon:10090",
) -> pd.DataFrame:
    source_ids = [cell_id.replace("-1", ".1") for cell_id in cell_ids]
    return pd.DataFrame(
        {
            "geo_cell_id": cell_ids,
            "orig.ident": [f"{species}-sample"] * len(cell_ids),
            "celltype": ["Pericyte"] * len(cell_ids),
            "cell_type": ["Pericyte"] * len(cell_ids),
            "cell_type_ontology_term_id": ["CL:0000669"] * len(cell_ids),
            "species": [species] * len(cell_ids),
            "source_organism_ontology_term_id": [organism] * len(cell_ids),
            "organism_ontology_term_id": [organism] * len(cell_ids),
            "assay_ontology_term_id": ["EFO:0030080"] * len(cell_ids),
        },
        index=pd.Index(source_ids, name="source_obs_name"),
    )


def test_load_geo_count_subset_uses_integer_counts_and_annotation_order(tmp_path):
    path = tmp_path / "counts.csv.gz"
    pd.DataFrame(
        {
            "": ["H1", "H2", "H3"],
            "cell-a-1": [1, 0, 9],
            "cell-b-1": [0, 3, 8],
            "not-annotated-1": [7, 7, 7],
        }
    ).to_csv(path, index=False, compression="gzip")
    mapping = pd.DataFrame(
        {"source_gene": ["H1", "H2"], "mouse_gene": ["m1", "m2"]}
    )
    annotations = _annotations("human", ["cell-b-1", "cell-a-1"])

    result = load_geo_count_subset(
        path, annotations, mapping, chunksize=1
    )

    assert result.obs_names.tolist() == ["cell-b.1", "cell-a.1"]
    assert result.var_names.tolist() == ["m1", "m2"]
    assert result.X.toarray().tolist() == [[0, 3], [1, 0]]
    assert np.issubdtype(result.X.dtype, np.integer)
    assert result.var["organism"].unique().tolist() == ["NCBITaxon:10090"]
    assert result.uns["eye_geo_import"]["integer_counts"] is True


def test_load_geo_count_subset_keeps_native_feature_organism(tmp_path):
    path = tmp_path / "counts.csv.gz"
    pd.DataFrame({"gene": ["H1", "H2"], "cell-a-1": [1, 2]}).to_csv(
        path, index=False, compression="gzip"
    )
    annotations = _annotations(
        "human", ["cell-a-1"], organism="NCBITaxon:9606"
    )

    result = load_geo_count_subset(path, annotations, mapping=None)

    assert result.var_names.tolist() == ["H1", "H2"]
    assert result.var["organism"].unique().tolist() == ["NCBITaxon:9606"]


def test_load_geo_count_subset_sums_duplicate_human_ensembl_targets(tmp_path):
    path = tmp_path / "counts.csv.gz"
    pd.DataFrame(
        {
            "gene": ["AAK1_n", "AAK1_p", "OTHER"],
            "cell-a-1": [1, 3, 7],
            "cell-b-1": [2, 4, 8],
        }
    ).to_csv(path, index=False, compression="gzip")
    mapping = pd.DataFrame(
        {
            "source_gene": ["AAK1_n", "AAK1_p"],
            "target_gene": ["ENSG1", "ENSG1"],
        }
    )
    annotations = _annotations(
        "macaque_fascicularis",
        ["cell-a-1", "cell-b-1"],
        organism="NCBITaxon:9606",
    )

    result = load_geo_count_subset(
        path,
        annotations,
        mapping,
        aggregate_duplicate_targets=True,
        target_id_type="ensembl",
    )

    assert result.var_names.tolist() == ["ENSG1"]
    assert result.X.toarray().tolist() == [[4], [6]]
    assert result.var["ensembl_gene_id"].tolist() == ["ENSG1"]
    assert result.uns["eye_geo_import"]["mapped_source_gene_rows"] == 2
    assert result.uns["eye_geo_import"]["aggregated_target_gene_rows"] == 1


def test_load_geo_count_subset_rejects_log_normalized_values(tmp_path):
    path = tmp_path / "counts.csv.gz"
    pd.DataFrame({"gene": ["H1"], "cell-a-1": [1.25]}).to_csv(
        path, index=False, compression="gzip"
    )
    mapping = pd.DataFrame(
        {"source_gene": ["H1"], "mouse_gene": ["m1"]}
    )

    with pytest.raises(ValueError, match="not an integer count matrix"):
        load_geo_count_subset(path, _annotations("human", ["cell-a-1"]), mapping)


def test_combine_eye_species_uses_only_shared_mouse_genes():
    human = ad.AnnData(
        X=sp.csr_matrix([[1, 2]]),
        obs=_annotations("human", ["human-a-1"]),
        var=pd.DataFrame(index=["m1", "m2"]),
    )
    mouse = ad.AnnData(
        X=sp.csr_matrix([[3, 4]]),
        obs=_annotations("mouse", ["mouse-a-1"]),
        var=pd.DataFrame(index=["m2", "m3"]),
    )

    result = combine_eye_species([human, mouse])

    assert result.shape == (2, 1)
    assert result.var_names.tolist() == ["m2"]
    assert result.X.toarray().tolist() == [[2], [3]]
    assert result.obs["species"].tolist() == ["human", "mouse"]


def test_align_eye_to_checkpoint_genes_keeps_complete_native_vocabulary():
    source = ad.AnnData(
        X=sp.csr_matrix([[3, 4], [0, 5]], dtype=np.int32),
        obs=_annotations(
            "human",
            ["human-a-1", "human-b-1"],
            organism="NCBITaxon:9606",
        ),
        var=pd.DataFrame(
            {
                "symbol": ["A", "C"],
                "organism": ["NCBITaxon:9606"] * 2,
                "ensembl_gene_id": ["ENSG1", "ENSG3"],
            },
            index=["ENSG1", "ENSG3"],
        ),
    )

    result = align_eye_to_checkpoint_genes(
        source,
        {
            "NCBITaxon:9606": ["ENSG1", "ENSG2", "ENSG3"],
        },
        min_checkpoint_genes=3,
        min_input_gene_overlap=2,
    )

    assert result.var_names.tolist() == ["ENSG1", "ENSG2", "ENSG3"]
    assert result.X.toarray().tolist() == [[3, 0, 4], [0, 0, 5]]
    assert result.var["organism"].unique().tolist() == ["NCBITaxon:9606"]
    provenance = result.uns["eye_native_checkpoint_alignment"]
    assert provenance["checkpoint_genes"] == 3
    assert provenance["preprocessed_gene_overlap"] == 2
    assert provenance["expressed_checkpoint_genes"] == 2


def test_combine_eye_native_species_builds_ordered_block_diagonal_matrix():
    mouse = ad.AnnData(
        X=sp.csr_matrix([[1, 2]], dtype=np.int32),
        obs=_annotations("mouse", ["mouse-a-1"]),
        var=pd.DataFrame(
            {
                "symbol": ["m1", "m2"],
                "organism": ["NCBITaxon:10090"] * 2,
                "ensembl_gene_id": ["M1", "M2"],
            },
            index=["M1", "M2"],
        ),
    )
    human = ad.AnnData(
        X=sp.csr_matrix([[3, 4, 5]], dtype=np.int32),
        obs=_annotations(
            "human", ["human-a-1"], organism="NCBITaxon:9606"
        ),
        var=pd.DataFrame(
            {
                "symbol": ["h1", "h2", "h3"],
                "organism": ["NCBITaxon:9606"] * 3,
                "ensembl_gene_id": ["H1", "H2", "H3"],
            },
            index=["H1", "H2", "H3"],
        ),
    )

    result = combine_eye_native_species(
        [human, mouse],
        organism_order=["NCBITaxon:10090", "NCBITaxon:9606"],
        min_genes_per_species=2,
    )

    assert result.var_names.tolist() == ["M1", "M2", "H1", "H2", "H3"]
    assert result.obs["species"].tolist() == ["mouse", "human"]
    assert result.X.toarray().tolist() == [
        [1, 2, 0, 0, 0],
        [0, 0, 3, 4, 5],
    ]
    assert result.uns["eye_cross_species"]["genes_by_species"] == {
        "mouse": 2,
        "human": 3,
    }


def test_combine_eye_ortholog_species_uses_union_and_audits_common_genes():
    human = ad.AnnData(
        X=sp.csr_matrix([[1, 2, 3]], dtype=np.int32),
        obs=_annotations("human", ["human-a-1"]),
        var=pd.DataFrame(index=["m1", "m2", "m3"]),
    )
    pig = ad.AnnData(
        X=sp.csr_matrix([[4, 5, 6]], dtype=np.int32),
        obs=_annotations("pig", ["pig-a-1"]),
        var=pd.DataFrame(index=["m2", "m3", "m4"]),
    )

    result = combine_eye_ortholog_species(
        [human, pig], min_genes_per_species=3, min_common_genes=2
    )

    assert result.var_names.tolist() == ["m1", "m2", "m3", "m4"]
    assert result.X.toarray().tolist() == [[1, 2, 3, 0], [0, 4, 5, 6]]
    assert result.uns["eye_cross_species"]["common_mouse_orthologs"] == 2
    assert result.uns["eye_cross_species"]["union_mouse_orthologs"] == 4


def test_combine_eye_target_species_uses_human_union_without_common_gene_gate():
    human = ad.AnnData(
        X=sp.csr_matrix([[1, 2]], dtype=np.int32),
        obs=_annotations(
            "human", ["human-a-1"], organism="NCBITaxon:9606"
        ),
        var=pd.DataFrame(index=["E1", "E2"]),
    )
    mouse = ad.AnnData(
        X=sp.csr_matrix([[3, 4]], dtype=np.int32),
        obs=_annotations(
            "mouse", ["mouse-a-1"], organism="NCBITaxon:9606"
        ),
        var=pd.DataFrame(index=["E3", "E4"]),
    )

    result = combine_eye_target_species(
        [human, mouse],
        target_organism_ontology_term_id="NCBITaxon:9606",
        feature_space="human_ensembl_union",
        min_genes_per_species=2,
    )

    assert result.var_names.tolist() == ["E1", "E2", "E3", "E4"]
    assert result.X.toarray().tolist() == [[1, 2, 0, 0], [0, 0, 3, 4]]
    assert result.uns["eye_cross_species"]["common_target_genes_audit_only"] == 0
    assert result.uns["eye_cross_species"]["union_target_genes"] == 4
    assert result.uns["eye_cross_species"]["combination_rule"] == (
        "union_zero_fill_no_intersection_filter"
    )


def test_align_eye_to_reference_genes_preserves_counts_and_reference_order():
    source = ad.AnnData(
        X=sp.csr_matrix([[2, 0, 3], [0, 0, 4], [1, 1, 0]], dtype=np.int32),
        obs=_annotations(
            "human", ["human-a-1", "human-b-1", "human-c-1"]
        ),
        var=pd.DataFrame(
            {"symbol": ["m2", "ambiguous", "m1"]},
            index=["m2", "ambiguous", "m1"],
        ),
    )
    reference = ad.AnnData(
        X=sp.csr_matrix((1, 5)),
        var=pd.DataFrame(
            {
                "symbol": ["unused", "m1", "ambiguous", "ambiguous", "m2"],
                "ensembl_gene_id": ["E0", "E1", "E2", "E3", "E4"],
                "organism": ["NCBITaxon:10090"] * 5,
            },
            index=["E0", "E1", "E2", "E3", "E4"],
        ),
    )

    result = align_eye_to_reference_genes(
        source, reference, min_valid_genes_id=2, min_nnz_genes=2
    )

    assert result.obs_names.tolist() == ["human-a.1", "human-c.1"]
    assert result.var_names.tolist() == ["E0", "E1", "E2", "E3", "E4"]
    assert result.X.toarray().tolist() == [
        [0, 3, 0, 0, 2],
        [0, 0, 0, 0, 1],
    ]
    provenance = result.uns["eye_cross_species"]["scprint_preprocessing"]
    assert provenance["registry_lookup"] is False
    assert provenance["pseudo_count_recovery"] is False
    assert provenance["min_nnz_genes"] == 2
    assert provenance["uniquely_mapped_input_symbols"] == 2


def test_align_eye_to_reference_genes_applies_dataset_gene_gate():
    source = ad.AnnData(
        X=sp.csr_matrix([[1]], dtype=np.int32),
        obs=_annotations("mouse", ["mouse-a-1"]),
        var=pd.DataFrame({"symbol": ["m1"]}, index=["m1"]),
    )
    reference = ad.AnnData(
        X=sp.csr_matrix((1, 1)),
        var=pd.DataFrame(
            {
                "symbol": ["m1"],
                "ensembl_gene_id": ["E1"],
                "organism": ["NCBITaxon:10090"],
            },
            index=["E1"],
        ),
    )

    with pytest.raises(ValueError, match="min_valid_genes_id"):
        align_eye_to_reference_genes(
            source,
            reference,
            min_valid_genes_id=2,
            min_nnz_genes=1,
        )


def test_align_eye_to_reference_genes_filters_cells_before_symbol_mapping():
    source = ad.AnnData(
        X=sp.csr_matrix([[1, 1]], dtype=np.int32),
        obs=_annotations("mouse", ["mouse-a-1"]),
        var=pd.DataFrame(
            {"symbol": ["m1", "not_in_reference"]},
            index=["m1", "not_in_reference"],
        ),
    )
    reference = ad.AnnData(
        X=sp.csr_matrix((1, 1)),
        var=pd.DataFrame(
            {
                "symbol": ["m1"],
                "ensembl_gene_id": ["E1"],
                "organism": ["NCBITaxon:10090"],
            },
            index=["E1"],
        ),
    )

    result = align_eye_to_reference_genes(
        source,
        reference,
        min_valid_genes_id=1,
        min_nnz_genes=2,
    )

    assert result.shape == (1, 1)
    assert result.X.toarray().tolist() == [[1]]


def test_write_eye_manifest_serializes_numpy_provenance(tmp_path):
    data = ad.AnnData(
        X=sp.csr_matrix([[1]], dtype=np.int64),
        obs=pd.DataFrame(
            {"species": ["mouse"], "celltype": ["Pericyte"]},
            index=["cell"],
        ),
        var=pd.DataFrame(index=["gene"]),
        uns={"eye_cross_species": {"mapped_genes": np.int64(1)}},
    )

    output = write_eye_manifest(data, tmp_path / "manifest.json")
    payload = json.loads(output.read_text())

    assert payload["provenance"]["mapped_genes"] == 1


def test_write_eye_orthology_audit_collects_species_provenance(tmp_path):
    data = ad.AnnData(
        X=sp.csr_matrix([[1]], dtype=np.int32),
        obs=_annotations("human", ["human-a-1"]),
        var=pd.DataFrame(index=["m1"]),
        uns={
            "eye_geo_import": {
                "source_gene_rows": np.int64(10),
                "mapped_gene_rows": np.int64(1),
                "integer_counts": True,
                "source_sha256": "abc",
            },
            "eye_orthology": {
                "input_rows": np.int64(4),
                "unique_pairs": np.int64(3),
                "conflicting_source_genes": np.int64(1),
                "conflicting_mouse_genes": np.int64(0),
                "retained_pairs": np.int64(2),
            },
        },
    )
    source = tmp_path / "human.h5ad"
    data.write_h5ad(source)

    output = write_eye_orthology_audit([source], tmp_path / "audit.tsv")
    audit = pd.read_csv(output, sep="\t")

    assert audit.loc[0, "species"] == "human"
    assert audit.loc[0, "orthology_mode"] == "strict_one_to_one"
    assert audit.loc[0, "conflicting_source_genes"] == 1
    assert audit.loc[0, "mapped_gene_rows_in_geo"] == 1


def test_write_eye_native_gene_audit_reports_species_proxy_and_gene_counts(tmp_path):
    data = ad.AnnData(
        X=sp.csr_matrix([[1, 0, 2]], dtype=np.int32),
        obs=_annotations(
            "macaque_fascicularis",
            ["macaque-a-1"],
            organism="NCBITaxon:9544",
        ),
        var=pd.DataFrame(
            {"organism": ["NCBITaxon:9544"] * 3},
            index=["R1", "R2", "R3"],
        ),
        uns={
            "eye_geo_import": {"source_gene_rows": 10, "source_sha256": "abc"},
            "eye_native_checkpoint_alignment": {
                "checkpoint_genes": 3,
                "preprocessed_gene_overlap": 3,
                "expressed_checkpoint_genes": 2,
                "count_sum": 3,
            },
        },
    )
    data.obs["source_organism_ontology_term_id"] = "NCBITaxon:9541"
    source = tmp_path / "macaque.h5ad"
    data.write_h5ad(source)

    output = write_eye_native_gene_audit([source], tmp_path / "native.tsv")
    audit = pd.read_csv(output, sep="\t")

    assert bool(audit.loc[0, "uses_supported_species_proxy"]) is True
    assert audit.loc[0, "checkpoint_genes"] == 3
    assert audit.loc[0, "expressed_checkpoint_genes"] == 2
