from pathlib import Path

import pandas as pd

from scprint2.evaluation.result_collection import (
    COMMON_BATCH_METRICS,
    COMMON_BIO_METRICS,
)
from scprint2.evaluation.result_collection import (
    aggregate_task3_paper_scores as aggregate,
)
from scprint2.evaluation.result_collection import (
    aggregate_task3_raw_common_scores as aggregate_raw_common,
)


def test_reference_raw_metrics_reconstruct_reported_task3_scores():
    reference = pd.read_csv(
        Path(__file__).parents[1]
        / "data"
        / "reference"
        / "cross_species_task3_paper_scores.csv"
    )
    reported = reference.set_index("Method")[
        ["Overall.Score", "Batch.Correction", "Bio.conservation"]
    ]
    raw = reference.drop(
        columns=["Overall.Score", "Batch.Correction", "Bio.conservation"]
    )
    reconstructed = aggregate(raw).set_index("Method")[reported.columns]

    # The published inputs are rounded to three decimals before reconstruction.
    assert (reported - reconstructed).abs().to_numpy().max() < 0.0025


def test_joint_aggregation_uses_one_shared_metric_scale():
    reference = pd.read_csv(
        Path(__file__).parents[1]
        / "data"
        / "reference"
        / "cross_species_task3_paper_scores.csv"
    )
    raw = reference.drop(
        columns=["Overall.Score", "Batch.Correction", "Bio.conservation"]
    )
    first = aggregate(raw)
    duplicated = pd.concat([raw, raw.iloc[[0]].assign(Method="new")], ignore_index=True)
    second = aggregate(duplicated)

    expected = first.loc[first["Method"] == "SATURN", "Overall.Score"].iloc[0]
    observed = second.loc[second["Method"] == "new", "Overall.Score"].iloc[0]
    assert observed == expected


def test_paper_anchored_aggregation_keeps_reference_scores_fixed():
    reference = pd.read_csv(
        Path(__file__).parents[1]
        / "data"
        / "reference"
        / "cross_species_task3_paper_scores.csv"
    )
    raw = reference.drop(
        columns=["Overall.Score", "Batch.Correction", "Bio.conservation"]
    )
    extended = pd.concat([raw, raw.iloc[[0]].assign(Method="new")], ignore_index=True)
    anchored = aggregate(extended, anchors=raw).set_index("Method")
    reconstructed = aggregate(raw).set_index("Method")

    pd.testing.assert_series_equal(
        anchored.loc["SATURN"], reconstructed.loc["SATURN"], check_names=False
    )


def test_raw_common_aggregation_does_not_rescale_or_change_denominators():
    reference = pd.read_csv(
        Path(__file__).parents[1]
        / "data"
        / "reference"
        / "cross_species_task3_paper_scores.csv"
    )
    raw = reference.drop(
        columns=["Overall.Score", "Batch.Correction", "Bio.conservation"]
    )
    result = aggregate_raw_common(raw).set_index("Method")
    saturn = raw.set_index("Method").loc["SATURN"]

    expected_batch = saturn[COMMON_BATCH_METRICS].mean()
    expected_bio = saturn[COMMON_BIO_METRICS].mean()
    assert result.loc["SATURN", "Raw.Batch.Mean"] == expected_batch
    assert result.loc["SATURN", "Raw.Bio.Mean"] == expected_bio
    assert result.loc["SATURN", "Raw.Overall"] == 0.4 * expected_batch + 0.6 * expected_bio
