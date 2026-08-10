"""Gene metadata helpers that do not require a live LaminDB registry."""

from typing import Any

import pandas as pd


def active_model_organisms(model: Any, obs: pd.DataFrame) -> list[str]:
    """Return model organisms actually represented in an input dataset."""
    column = "organism_ontology_term_id"
    if column not in obs:
        raise KeyError(f"The input observations have no {column!r} column")
    organisms = list(pd.unique(obs[column].astype(str)))
    unsupported = sorted(set(organisms) - set(model.organisms))
    if unsupported:
        raise ValueError(f"The model does not support organisms {unsupported}")
    return organisms


def model_gene_dataframe(
    model: Any, input_var: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the full Collator gene table without querying LaminDB."""
    if input_var is not None and "organism" in input_var:
        if input_var["organism"].isna().any():
            raise ValueError("The input gene table contains missing organism values")
        return input_var[["organism"]].copy()

    organisms = list(model.organisms)
    genes = model._genes
    if isinstance(genes, dict):
        missing = [organism for organism in organisms if organism not in genes]
        if missing:
            raise KeyError(f"The model checkpoint has no genes for {missing}")
        genes_by_organism = {organism: genes[organism] for organism in organisms}
    else:
        if len(organisms) != 1:
            raise ValueError(
                "A flat checkpoint gene vocabulary is only unambiguous for one organism"
            )
        genes_by_organism = {organisms[0]: genes}

    return pd.concat(
        [
            pd.DataFrame(
                {"organism": organism},
                index=pd.Index(organism_genes, name="ensembl_gene_id"),
            )
            for organism, organism_genes in genes_by_organism.items()
        ]
    )


def set_collator_organism_ids(
    collator: Any, organisms: list[str], org_to_id: dict[str, int] | None = None
) -> None:
    """Set IDs omitted by scDataLoader when a prebuilt gene table is supplied."""
    collator.organism_ids = {
        org_to_id[organism] if org_to_id is not None else organism
        for organism in organisms
    }
