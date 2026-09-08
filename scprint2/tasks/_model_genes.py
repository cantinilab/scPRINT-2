"""Gene metadata helpers that do not require a live LaminDB registry."""

from typing import Any

import numpy as np
import pandas as pd


class OrganismBlockCollator:
    """Slice block-diagonal multi-organism expression before scDataLoader."""

    def __init__(self, collator: Any, masks: dict[Any, np.ndarray]) -> None:
        self.collator = collator
        self.masks = masks

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "toarray"):
            return np.asarray(value.toarray()).squeeze()
        return np.asarray(value)

    def __call__(self, batch: list[dict[str, Any]]) -> Any:
        sliced = []
        for element in batch:
            organism_id = element[self.collator.organism_name]
            if organism_id not in self.masks:
                sliced.append(element)
                continue
            mask = self.masks[organism_id]
            current = dict(element)
            expression = self._array(current["X"])
            if expression.ndim != 1 or expression.shape[0] != len(mask):
                raise ValueError(
                    "Multi-organism expression does not match the input gene table"
                )
            current["X"] = expression[mask]
            if "knn_cells" in current:
                neighbors = self._array(current["knn_cells"])
                if neighbors.ndim != 2 or neighbors.shape[1] != len(mask):
                    raise ValueError(
                        "Multi-organism KNN expression does not match the gene table"
                    )
                current["knn_cells"] = neighbors[:, mask]
            sliced.append(current)
        return self.collator(sliced)


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
    """Build the global Collator gene table without querying LaminDB.

    The model gene encoder uses one vocabulary concatenated across organisms.  Even
    when an input contains a single organism, the preceding organisms must remain in
    the table so that ``Collator.start_idx`` keeps the checkpoint's global offsets.
    For organisms present in the input, retain the complete input gene table so the
    collator can construct a mask matching the expression matrix width.
    """
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

    input_by_organism: dict[str, pd.DataFrame] = {}
    if input_var is not None and "organism" in input_var:
        if input_var["organism"].isna().any():
            raise ValueError("The input gene table contains missing organism values")
        for organism, frame in input_var[["organism"]].groupby(
            "organism", sort=False, observed=True
        ):
            input_by_organism[str(organism)] = frame

    for organism, frame in input_by_organism.items():
        if organism not in genes_by_organism:
            continue
        expected_genes = genes_by_organism[organism]
        expected_set = set(expected_genes)
        observed_genes = [gene for gene in frame.index if gene in expected_set]
        if observed_genes != expected_genes:
            missing = len(expected_set - set(observed_genes))
            raise RuntimeError(
                "Input gene vocabulary is incompatible with the checkpoint for "
                f"{organism}: {missing} checkpoint genes are missing or the common "
                "genes are not in checkpoint order. Re-preprocess the dataset or "
                "explicitly resize the model vocabulary before inference."
            )

    frames = []
    for organism in organisms:
        if organism in input_by_organism:
            frames.append(input_by_organism[organism])
        else:
            frames.append(
                pd.DataFrame(
                    {"organism": organism},
                    index=pd.Index(
                        genes_by_organism[organism], name="ensembl_gene_id"
                    ),
                )
            )
    return pd.concat(frames)


def expected_model_gene_offsets(model: Any) -> dict[str, int]:
    """Return the checkpoint's global gene offset for every organism."""
    organisms = list(model.organisms)
    genes = model._genes
    if not isinstance(genes, dict):
        if len(organisms) != 1:
            raise ValueError(
                "A flat checkpoint gene vocabulary is only unambiguous for one organism"
            )
        return {organisms[0]: 0}

    offsets: dict[str, int] = {}
    offset = 0
    for organism in organisms:
        if organism not in genes:
            raise KeyError(f"The model checkpoint has no genes for {organism}")
        offsets[organism] = offset
        offset += len(genes[organism])
    return offsets


def set_collator_organism_ids(
    collator: Any, organisms: list[str], org_to_id: dict[str, int] | None = None
) -> None:
    """Set IDs omitted by scDataLoader when a prebuilt gene table is supplied."""
    collator.organism_ids = {
        org_to_id[organism] if org_to_id is not None else organism
        for organism in organisms
    }


def validate_collator_gene_offsets(
    collator: Any,
    model: Any,
    organisms: list[str],
    org_to_id: dict[str, int] | None = None,
) -> None:
    """Fail before inference if a collator would address the wrong model genes."""
    expected = expected_model_gene_offsets(model)
    for organism in organisms:
        key = org_to_id[organism] if org_to_id is not None else organism
        actual = collator.start_idx.get(key)
        if actual != expected[organism]:
            raise RuntimeError(
                "Collator gene vocabulary is incompatible with the checkpoint: "
                f"{organism} starts at {actual}, expected {expected[organism]}. "
                "Keep all checkpoint organisms in model order when building genedf."
            )


def collator_for_organism_blocks(
    collator: Any,
    input_var: pd.DataFrame,
    organisms: list[str],
    org_to_id: dict[str, int] | None = None,
) -> Any:
    """Wrap a collator when AnnData stores native vocabularies in gene blocks."""
    if "organism" not in input_var or len(organisms) < 2:
        return collator
    var_organisms = input_var["organism"].astype(str).to_numpy()
    masks: dict[Any, np.ndarray] = {}
    for organism in organisms:
        key = org_to_id[organism] if org_to_id is not None else organism
        mask = var_organisms == organism
        expected_width = len(collator.accepted_genes[key])
        if int(mask.sum()) != expected_width:
            raise RuntimeError(
                f"Input gene block for {organism} has {int(mask.sum())} genes; "
                f"the collator expects {expected_width}"
            )
        masks[key] = mask
    return OrganismBlockCollator(collator, masks)
