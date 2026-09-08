import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from scprint2.plotting.openproblems import (
    BATCH_DISPLAY_NAMES,
    BATCH_HIDDEN_METHODS,
    BATCH_METHOD_ORDER,
    CLASSIFICATION_DATASET_LABELS,
    CLASSIFICATION_DISPLAY_NAMES,
    CLASSIFICATION_METHOD_ORDER,
    _fixed_denominator_mean,
    generate_openproblems_plots,
    load_batch_tables,
    load_classification_tables,
    plot_score_distribution,
    prepare_scores,
)


def test_prepare_scores_keeps_missing_scores_when_excluded():
    raw_scores = pd.DataFrame(
        {"dataset_a": [1.0, np.nan], "dataset_b": [0.5, 0.25]},
        index=["method_a", "method_b"],
    )

    scores = prepare_scores(raw_scores, use_zscore=False, exclude_missing=True)

    assert np.isnan(scores.loc["method_b", "dataset_a"])
    assert scores.loc["method_a", "dataset_b"] == 0.5


def test_prepare_scores_penalizes_missing_scores_when_not_excluded():
    raw_scores = pd.DataFrame(
        {"dataset_a": [1.0, np.nan], "dataset_b": [0.5, 0.25]},
        index=["method_a", "method_b"],
    )

    scores = prepare_scores(raw_scores, use_zscore=False, exclude_missing=False)

    assert scores.loc["method_b", "dataset_a"] == 0.0
    assert scores.loc["method_b"].mean() == 0.125


def test_prepare_scores_keeps_real_zero_distinct_from_missing_value():
    raw_scores = pd.DataFrame(
        {"dataset_a": [0.0, np.nan], "dataset_b": [0.5, 0.25]},
        index=["method_with_zero", "method_with_missing"],
    )

    scores = prepare_scores(raw_scores, use_zscore=False, exclude_missing=True)

    assert scores.loc["method_with_zero", "dataset_a"] == 0.0
    assert np.isnan(scores.loc["method_with_missing", "dataset_a"])


def test_prepare_scores_zscores_by_dataset_and_keeps_constant_columns_zero():
    raw_scores = pd.DataFrame(
        {"variable": [1.0, 3.0], "constant": [0.5, 0.5]},
        index=["method_a", "method_b"],
    )

    scores = prepare_scores(raw_scores, use_zscore=True, exclude_missing=True)

    assert scores.loc["method_a", "variable"] == -1.0
    assert scores.loc["method_b", "variable"] == 1.0
    assert scores.loc["method_a", "constant"] == 0.0
    assert scores.loc["method_b", "constant"] == 0.0


def test_fixed_denominator_mean_preserves_missing_metric_penalty():
    metrics = pd.DataFrame(
        {"method_a": [1.0, np.nan], "method_b": [np.nan, np.nan]},
        index=["metric_a", "metric_b"],
    )

    means = _fixed_denominator_mean(metrics, ["metric_a", "metric_b"])

    assert means["method_a"] == 0.5
    assert np.isnan(means["method_b"])


def test_batch_total_weights_biology_at_sixty_percent(tmp_path):
    rows = []
    for dataset in CLASSIFICATION_DATASET_LABELS:
        for metric in ["ari", "nmi", "isolated_label_asw", "clisi", "asw_label"]:
            rows.append({"dataset": dataset, "metric": metric, "scPRINT-1": 1.0})
        for metric in ["pcr", "graph_connectivity", "asw_batch", "ilisi", "kbet"]:
            rows.append({"dataset": dataset, "metric": metric, "scPRINT-1": 0.0})
    source = tmp_path / "batch.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    tables = load_batch_tables(source)

    assert (tables["Bio conservation"].loc["scPRINT-1"] == 1.0).all()
    assert (tables["Batch correction"].loc["scPRINT-1"] == 0.0).all()
    assert (tables["Total"].loc["scPRINT-1"] == 0.6).all()


def test_classification_tables_hide_uce_and_keep_both_scimilarity_variants(tmp_path):
    rows = []
    for score in ["macro_f1", "weighted", "accuracy"]:
        for dataset in CLASSIFICATION_DATASET_LABELS:
            rows.append(
                {
                    "score": score,
                    "method": dataset,
                    "uce": 0.1,
                    "scimilarity": 0.2,
                    "scimilarity_knn": 0.3,
                }
            )
    source = tmp_path / "classification.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    tables = load_classification_tables(source)

    assert list(tables["accuracy"].index) == ["scimilarity_knn", "scimilarity"]


def test_requested_method_order_and_display_names():
    assert CLASSIFICATION_METHOD_ORDER[:2] == ["majority_vote", "random_labels"]
    assert CLASSIFICATION_DISPLAY_NAMES["scimilarity_knn"] == "SCimilarity (kNN)"
    assert (
        CLASSIFICATION_DISPLAY_NAMES["scimilarity"]
        == "SCimilarity (kNN 9M atlas)"
    )
    assert (
        CLASSIFICATION_DISPLAY_NAMES["scPRINT-1 (zero-shot)"]
        == "scPRINT-1 (zero-shot)"
    )
    assert (
        CLASSIFICATION_DISPLAY_NAMES["scPRINT-2 (zero-shot)"]
        == "scPRINT-2 (zero-shot)"
    )
    assert "geneformer" not in BATCH_METHOD_ORDER
    assert "geneformer" in BATCH_HIDDEN_METHODS
    assert BATCH_DISPLAY_NAMES["scgpt_zeroshot"] == "scGPT"
    assert BATCH_DISPLAY_NAMES["scPRINT-1"] == "scPRINT-1 (zero-shot)"
    assert (
        BATCH_DISPLAY_NAMES["scPRINT-2 ZS cell_type"] == "scPRINT-2 (zero-shot)"
    )


def test_plot_writes_only_png_and_does_not_draw_zero_dataset_dot(
    tmp_path, monkeypatch
):
    raw_scores = pd.DataFrame(
        {
            "dkd": [0.0, 0.8],
            "gtex_v9": [0.4, 0.7],
            "hypomap": [0.5, 0.6],
            "mouse_pancreas_atlas": [0.6, 0.5],
        },
        index=["method_a", "method_b"],
    )
    plotted_y_values = []
    original_scatter = Axes.scatter

    def capture_scatter(self, x, y, *args, **kwargs):
        plotted_y_values.append(np.asarray(y))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)

    plot_score_distribution(
        raw_scores,
        score_name="Accuracy",
        title="Classification",
        filename="scores",
        output_dir=tmp_path,
        dataset_labels=CLASSIFICATION_DATASET_LABELS,
        use_zscore=False,
        exclude_missing=False,
        separators=set(),
        figure_size=(8, 6),
        display_names={},
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["scores.png"]
    assert 0.0 not in plotted_y_values[0]
    assert np.isclose(plotted_y_values[-1][0], 0.375)


def test_generate_plots_uses_penalized_raw_and_zscore_variants(
    tmp_path, monkeypatch
):
    classification = {
        name: pd.DataFrame()
        for name in ("f1_weighted", "f1_micro", "f1_macro", "accuracy")
    }
    batch = {
        name: pd.DataFrame()
        for name in ("Total", "Bio conservation", "Batch correction")
    }
    calls = []

    monkeypatch.setattr(
        "scprint2.plotting.openproblems.load_classification_tables",
        lambda path: classification,
    )
    monkeypatch.setattr(
        "scprint2.plotting.openproblems.load_batch_tables", lambda path: batch
    )
    monkeypatch.setattr(
        "scprint2.plotting.openproblems.plot_score_distribution",
        lambda *args, **kwargs: calls.append(
            (kwargs["use_zscore"], kwargs["exclude_missing"])
        ),
    )

    count = generate_openproblems_plots(tmp_path, tmp_path / "out")

    assert count == 14
    assert set(calls) == {
        (False, False),
        (True, False),
    }
    assert len(calls) == 14
