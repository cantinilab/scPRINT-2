from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_pandora_task3_matched_umaps.py"
SPEC = importlib.util.spec_from_file_location("plot_pandora_umaps", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_collapse_predictions_keeps_most_abundant_labels() -> None:
    values = pd.Series(["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"])
    collapsed, metadata = MODULE.collapse_predictions(values, top_n=2)

    assert set(collapsed) == {"A", "B", "Other predictions (2 labels)"}
    assert metadata["kept_categories"] == ["A", "B"]
    assert metadata["other_cells"] == 4


def test_converted_preserves_unmapped_ontology_ids() -> None:
    values = pd.Series(["CL:1", "CL:2", None])
    result = MODULE.converted(values, {"CL:1": "type one", "unknown": "Unknown"})

    assert result.tolist() == ["type one", "CL:2", "Unknown"]


def test_prediction_palette_marks_collapsed_labels_gray() -> None:
    colors = MODULE.categorical_palette(
        ["type one", "Other predictions (3 labels)"], prediction=True
    )

    assert colors["Other predictions (3 labels)"] == (0.65, 0.65, 0.65, 1.0)
