import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks/scPRINT-2-repro-notebooks"
RESULT_ROOT = (
    REPO_ROOT
    / "data/results/cross_species_embedding/eye4_human_target_hybrid_20260824"
)
NOTEBOOKS = (
    NOTEBOOK_ROOT / "cross-multi-species-embbedding.ipynb",
    NOTEBOOK_ROOT / "fine_tuning_multi_cross_species_emb_mmd.ipynb",
)


def _notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_ines_notebooks_match_final_human_target_run_contract() -> None:
    zero_code = _notebook_code(NOTEBOOKS[0])
    ft_code = _notebook_code(NOTEBOOKS[1])
    combined = zero_code + ft_code

    assert "eye4_human_target_hybrid_20260824" in combined
    assert "eye4_task3_corrected_20260823" not in combined
    assert "eye4_mouse_ortholog_scprint_preprocessed.h5ad" not in combined
    assert "align_checkpoint_genes=True" in zero_code
    assert "all_token_keys_except_organism" in zero_code
    assert "all-tokens-except-organism-PCA50" in zero_code
    assert "mmd_scale=0.03" in ft_code
    assert "include_organism_classification=False" in ft_code
    assert "require_known_train_labels=True" in ft_code
    assert "TranscriptFormer-Metazoa-human-target" in zero_code
    assert "Random-seed42" in zero_code


def test_final_eye4_notebooks_are_fully_executed_without_errors() -> None:
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert all(cell["execution_count"] is not None for cell in code_cells), path
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert errors == [], path


def test_six_method_scores_and_token_policy_match_notebook() -> None:
    scores = pd.read_csv(
        RESULT_ROOT
        / "scib113_all_tokens_except_organism/eye4_scib113_human_target_six_methods_scores.csv"
    )
    score_metadata = json.loads(
        (
            RESULT_ROOT
            / "scib113_all_tokens_except_organism/eye4_scib113_human_target_six_methods_scores.metadata.json"
        ).read_text()
    )
    token_metadata = json.loads(
        (
            RESULT_ROOT
            / "models/scprint2_zero_shot_small_v2_human_target_all_tokens_except_organism_pca50.metadata.json"
        ).read_text()
    )
    figure_metadata = json.loads(
        (
            RESULT_ROOT
            / "figures/scored_umaps_all_tokens_except_organism/eye4_scored_umaps.metadata.json"
        ).read_text()
    )

    assert scores.shape == (6, 20)
    assert score_metadata["versions"]["scib"] == "1.1.3"
    assert len(score_metadata["embeddings"]) == 6
    assert len(figure_metadata["methods"]) == 6
    assert token_metadata["token_policy"] == "all-except-organism"
    assert token_metadata["pca_dimension"] == 50
    assert "assay_ontology_term_id" in token_metadata["selected_tokens"]
    assert "organism_ontology_term_id" not in token_metadata["selected_tokens"]
