from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts/fetch_eye4_biomart_orthologs.py"
    spec = importlib.util.spec_from_file_location("fetch_eye4_biomart", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_biomart_query_requests_mouse_and_source_gene_identifiers():
    query = _module().biomart_query("mfascicularis")

    assert 'Dataset name="mmusculus_gene_ensembl"' in query
    assert 'Attribute name="external_gene_name"' in query
    assert 'Attribute name="mfascicularis_homolog_associated_gene_name"' in query
    assert 'Attribute name="mfascicularis_homolog_orthology_type"' in query
    assert 'Attribute name="mfascicularis_homolog_perc_id"' in query
    assert 'formatter="TSV" header="0"' in query


def test_macaque_ortholog_query_uses_the_geo_source_species():
    module = _module()

    assert (
        module.SPECIES_CONFIG["macaque_fascicularis"]["native_dataset"]
        == "mmulatta_gene_ensembl"
    )
    assert (
        module.SPECIES_CONFIG["macaque_fascicularis"]["ortholog_prefix"]
        == "mfascicularis"
    )
