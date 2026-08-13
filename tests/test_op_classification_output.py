import json

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad

from scripts.op_classification_output import save_classification_output


def test_save_classification_output_keeps_reusable_per_cell_data(tmp_path):
    embedded = AnnData(
        obs=pd.DataFrame(
            {
                "cell_type": ["alpha", "beta"],
                "cell_type_ontology_term_id": ["CL:1", "CL:2"],
                "donor_id": ["train", "test"],
                "classification_held_out": [False, True],
                "pred_cell_type_ontology_term_id_direct": ["CL:1", "CL:2"],
                "unused": [1, 2],
            },
            index=["cell-a", "cell-b"],
        )
    )
    embedded.obsm["scprint_emb"] = np.ones((2, 3), dtype=np.float32)
    embedded.obsm["classification_logits"] = np.asarray(
        [[0.9, 0.1], [0.2, 0.8]], dtype=np.float32
    )
    embedded.obsm["unrelated"] = np.zeros((2, 1), dtype=np.float32)
    embedded.uns["classification_logit_labels"] = ["CL:1", "CL:2"]

    output_path = tmp_path / "classification_output.h5ad"
    save_classification_output(
        embedded,
        output_path,
        metadata={"label_decoders": {"cell_type": {0: "CL:1", 1: "CL:2"}}},
    )

    saved = read_h5ad(output_path)
    assert saved.obs_names.tolist() == ["cell-a", "cell-b"]
    assert "unused" not in saved.obs
    assert set(saved.obsm) == {"classification_logits", "scprint_emb"}
    np.testing.assert_array_equal(
        saved.obsm["classification_logits"],
        embedded.obsm["classification_logits"],
    )
    assert saved.uns["classification_logit_labels"].tolist() == ["CL:1", "CL:2"]
    metadata = json.loads(output_path.with_suffix(".meta.json").read_text())
    assert metadata["n_cells"] == 2
    assert metadata["label_decoders"]["cell_type"]["0"] == "CL:1"
    assert metadata["classification_logit_labels"] == ["CL:1", "CL:2"]
