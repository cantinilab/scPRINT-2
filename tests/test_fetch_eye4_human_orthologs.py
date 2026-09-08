from scripts.fetch_eye4_human_orthologs import human_target_biomart_query


def test_human_target_biomart_query_uses_human_dataset_and_source_attributes():
    query = human_target_biomart_query("sscrofa")

    assert 'Dataset name="hsapiens_gene_ensembl"' in query
    assert 'Attribute name="ensembl_gene_id"' in query
    assert 'Attribute name="sscrofa_homolog_ensembl_gene"' in query
    assert 'Attribute name="sscrofa_homolog_orthology_confidence"' in query
