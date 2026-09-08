import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
NOTEBOOKS = [
    REPO_ROOT / "notebooks/scPRINT-2-repro-notebooks/cross-species-embbedding.ipynb",
    REPO_ROOT
    / "notebooks/scPRINT-2-repro-notebooks/fine_tuning_cross_species_emb_mmd.ipynb",
    REPO_ROOT / "notebooks/scPRINT-2-repro-notebooks/batch_corr_op.ipynb",
    REPO_ROOT / "notebooks/scPRINT-2-repro-notebooks/batch_corr_op ft.ipynb",
    REPO_ROOT / "notebooks/scPRINT-2-repro-notebooks/batch_corr_op v1.ipynb",
    REPO_ROOT
    / "notebooks/scPRINT-2-repro-notebooks/cross-species-embedding-pandora-lung.ipynb",
    REPO_ROOT
    / "notebooks/scPRINT-2-repro-notebooks/cross-multi-species-embbedding.ipynb",
    REPO_ROOT
    / "notebooks/scPRINT-2-repro-notebooks/fine_tuning_multi_cross_species_emb_mmd.ipynb",
    REPO_ROOT / "figures/plot_results.ipynb",
]

NOTEBOOK_REQUIREMENTS = {
    NOTEBOOKS[0]: {
        "markdown": (
            "## Data audit",
            "## Optional inference",
            "## Artifact validation",
            "## UMAP diagnostics",
            "## Detailed scIB scores",
            "## Interpretation boundary",
        ),
        "code": (
            "load_task3_dataset",
            "summarize_task3_dataset",
            "load_task3_checkpoint",
            "prepare_task3_dataset",
            "embed_task3",
            "save_task3_embedding",
            "task3_fresh_scored_embeddings_umap_overview.png",
        ),
    },
    NOTEBOOKS[1]: {
        "markdown": (
            "## Training data audit",
            "## MMD configuration",
            "## Optional fine-tuning",
            "## Training provenance",
            "## UMAP diagnostics",
            "## Detailed scIB scores",
            "## Interpretation boundary",
        ),
        "code": (
            "load_task3_dataset",
            "summarize_task3_dataset",
            "load_task3_checkpoint",
            "prepare_task3_dataset",
            "finetune_task3_checkpoint",
            "embed_task3",
            "save_task3_embedding",
            "scprint2_ft__organism.png",
        ),
    },
    NOTEBOOKS[2]: {
        "markdown": (
            "## Dataset and held-out audit",
            "## Optional embedding",
            "## Classification details",
            "## scIB breakdown",
            "## Comparison",
            "## Interpretation boundary",
        ),
        "code": (
            "load_openproblems_dataset",
            "summarize_openproblems_dataset",
            "load_model",
            "embed_all",
            "classification_scores",
            "select_embedding_view",
        ),
    },
    NOTEBOOKS[3]: {
        "markdown": (
            "## Dataset and held-out audit",
            "## MMD configuration",
            "## Optional fine-tuning",
            "## Classification details",
            "## scIB breakdown",
            "## Comparison",
            "## Interpretation boundary",
        ),
        "code": (
            "load_openproblems_dataset",
            "summarize_openproblems_dataset",
            "load_model",
            "finetune_openproblems_model",
            "embed_all",
            "classification_scores",
        ),
    },
    NOTEBOOKS[4]: {
        "markdown": (
            "## Dataset and held-out audit",
            "## Optional embedding",
            "## Classification details",
            "## scIB breakdown",
            "## Comparison",
            "## Interpretation boundary",
        ),
        "code": (
            "load_openproblems_v1_dataset",
            "summarize_openproblems_v1_dataset",
            "load_scprint1_model",
            "embed_scprint1",
            "classification_scores",
        ),
    },
    NOTEBOOKS[5]: {
        "markdown": (
            "## Annotation gate",
            "## Orthology audit",
            "## Optional preparation",
            "## Score breakdown",
            "## UMAP diagnostics",
            "## Interpretation boundary",
        ),
        "code": (
            "summarize_pandora_annotations",
            "summarize_pandora_orthology",
            "filter_h5ad_by_obs_labels",
            "build_benchmark_data",
            "pandora_task3_matched_umap_overview.png",
        ),
    },
    NOTEBOOKS[6]: {
        "markdown": (
            "## Protocol contract",
            "## Annotation gate",
            "## Orthology audit",
            "## Optional zero-shot inference",
            "## Detailed scIB 1.1.3 scores",
            "## UMAP diagnostics",
            "## Interpretation boundary",
        ),
        "code": (
            "extract_eye_annotations",
            "load_task3_dataset",
            "summarize_task3_dataset",
            "load_task3_checkpoint",
            "prepare_task3_dataset",
            "embed_task3",
            "save_task3_embedding",
            "eye4_scored_umaps_overview.png",
        ),
    },
    NOTEBOOKS[7]: {
        "markdown": (
            "## Protocol contract",
            "## Training data audit",
            "## MMD configuration",
            "## Optional fine-tuning",
            "## Training provenance",
            "## Detailed scIB 1.1.3 scores",
            "## UMAP diagnostics",
            "## Interpretation boundary",
        ),
        "code": (
            "load_task3_dataset",
            "summarize_task3_dataset",
            "load_task3_checkpoint",
            "prepare_task3_dataset",
            "finetune_task3_checkpoint",
            "embed_task3",
            "save_task3_embedding",
            "mmd_scale=0.03",
        ),
    },
    NOTEBOOKS[8]: {
        "markdown": (
            "## Inspect source tables",
            "## Missingness check",
            "## Raw-score figures",
            "## Z-score sensitivity",
            "## Reporting rule",
        ),
        "code": (
            "load_classification_tables",
            "load_batch_tables",
            "prepare_scores",
            "generate_openproblems_plots",
            "classification_f1_macro.png",
            "batch_total.png",
        ),
    },
}


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_explanatory_notebooks_have_markdown_and_no_local_definitions():
    for path in NOTEBOOKS:
        notebook = _load_notebook(path)
        markdown_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        ]
        assert len(markdown_cells) >= 7, path
        assert len(notebook["cells"]) >= 16, path

        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            definitions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            ]
            assert definitions == [], (path, cell_index, definitions)


def test_explanatory_notebooks_expose_scientific_stages_and_diagnostics():
    for path, requirements in NOTEBOOK_REQUIREMENTS.items():
        notebook = _load_notebook(path)
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        missing_markdown = [
            marker for marker in requirements["markdown"] if marker not in markdown
        ]
        missing_code = [marker for marker in requirements["code"] if marker not in code]
        assert missing_markdown == [], (path, missing_markdown)
        assert missing_code == [], (path, missing_code)


def test_explanatory_notebooks_load_reusable_package_code():
    task3_zero = NOTEBOOKS[0].read_text(encoding="utf-8")
    task3_ft = NOTEBOOKS[1].read_text(encoding="utf-8")
    openproblems = "\n".join(
        path.read_text(encoding="utf-8") for path in [*NOTEBOOKS[2:5], NOTEBOOKS[8]]
    )
    pandora = NOTEBOOKS[5].read_text(encoding="utf-8")
    eye_zero = NOTEBOOKS[6].read_text(encoding="utf-8")
    eye_ft = NOTEBOOKS[7].read_text(encoding="utf-8")

    assert "scprint2.benchmark.task3" in task3_zero
    assert "scprint2.evaluation.task3_scib" in task3_zero
    assert "scprint2.benchmark.task3" in task3_ft
    assert "scprint2.evaluation.task3_scib" in task3_ft
    assert "scprint2.benchmark.openproblems" in openproblems
    assert "scprint2.benchmark.openproblems_v1" in openproblems
    assert "scprint2.plotting.openproblems" in openproblems
    assert "filter_h5ad_by_obs_labels" in pandora
    assert "build_benchmark_data" in pandora
    assert "scprint2.preprocessing.eye" in eye_zero
    assert "scprint2.benchmark.task3" in eye_zero
    assert "scprint2.evaluation.task3_scib" in eye_zero
    assert "scprint2.benchmark.task3" in eye_ft


def test_explanatory_notebooks_do_not_hide_workflow_in_one_run_call():
    forbidden_calls = {
        "run_task3(",
        "run_openproblems_benchmark(",
        "run_openproblems_v1_benchmark(",
        "run_pandora_benchmark(",
    }
    for path in NOTEBOOKS:
        notebook = _load_notebook(path)
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        hits = sorted(call for call in forbidden_calls if call in code)
        assert hits == [], (path, hits)


def test_explanatory_notebooks_do_not_expose_shell_or_cli_execution():
    forbidden = {
        "subprocess",
        "sbatch",
        "srun",
        "slurm",
        "scripts/",
        "scripts\\\\",
        "os.system",
        "python -m",
    }
    for path in NOTEBOOKS:
        notebook = _load_notebook(path)
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            normalized = source.casefold()
            assert not source.lstrip().startswith(("!", "%run")), (
                path,
                cell_index,
            )
            hits = sorted(token for token in forbidden if token in normalized)
            assert hits == [], (path, cell_index, hits)


def test_executed_notebooks_have_no_error_outputs():
    for path in NOTEBOOKS:
        notebook = _load_notebook(path)
        errors = [
            output
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert errors == [], path
