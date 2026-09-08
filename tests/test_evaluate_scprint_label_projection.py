import numpy as np

from scprint2.evaluation.label_projection import (
    apply_mapping,
    evaluate,
    fit_direct_jaccard_mapping,
    fit_iterative_hungarian_mapping,
    hierarchy_correct,
    score_predictions,
    vote_neighbor_labels,
)


def test_direct_jaccard_allows_many_predictions_to_one_label():
    predicted = np.asarray(["X", "X", "Y", "Y", "Z"])
    truth = np.asarray(["A", "A", "A", "A", "B"])

    mapping = fit_direct_jaccard_mapping(predicted, truth)

    assert mapping == {"X": "A", "Y": "A", "Z": "B"}


def test_iterative_hungarian_competes_before_reusing_truth_labels():
    predicted = np.asarray(["X", "X", "X", "Y", "Y", "Y"])
    truth = np.asarray(["A", "A", "A", "A", "A", "B"])

    direct = fit_direct_jaccard_mapping(predicted, truth)
    hungarian = fit_iterative_hungarian_mapping(predicted, truth)

    assert direct["X"] == "A"
    assert direct["Y"] == "A"
    assert hungarian["X"] == "A"
    assert hungarian["Y"] == "B"


def test_hierarchy_first_maps_leaf_to_most_specific_available_parent():
    predicted = np.asarray(["leaf", "other", "other"])
    truth = np.asarray(["broad", "narrow", "narrow"])
    descendants = {
        "broad": {"leaf", "other"},
        "narrow": {"leaf"},
    }

    mapping = fit_direct_jaccard_mapping(
        predicted,
        truth,
        descendants=descendants,
    )

    assert mapping["leaf"] == "narrow"


def test_hierarchy_scoring_accepts_descendants_but_exact_f1_does_not():
    predicted = np.asarray(["leaf-a", "wrong"])
    truth = np.asarray(["parent", "leaf-b"])
    descendants = {"parent": {"leaf-a", "leaf-b"}}

    correct = hierarchy_correct(predicted, truth, descendants)
    scores = score_predictions(predicted, truth, descendants)

    np.testing.assert_array_equal(correct, [True, False])
    assert scores["accuracy_exact"] == 0.0
    assert scores["accuracy_hierarchy"] == 0.5


def test_apply_mapping_leaves_unseen_test_prediction_unchanged():
    predicted = np.asarray(["X", "unseen"])

    mapped = apply_mapping(predicted, {"X": "A"})

    np.testing.assert_array_equal(mapped, ["A", "unseen"])


def test_neighbor_vote_uses_requested_prefix_and_deterministic_ties():
    train_labels = np.asarray(["B", "A", "B", "A", "C"])
    neighbor_indices = np.asarray([[0, 1, 2, 3, 4]])

    ten_style = vote_neighbor_labels(
        neighbor_indices,
        train_labels,
        n_neighbors=3,
    )
    fifty_style = vote_neighbor_labels(
        neighbor_indices,
        train_labels,
        n_neighbors=4,
    )

    np.testing.assert_array_equal(ten_style, ["B"])
    np.testing.assert_array_equal(fifty_style, ["A"])


def test_evaluate_preserves_per_cell_mapping_predictions():
    arrays = {
        "truth": np.asarray(["A", "B", "A", "B"]),
        "held_out": np.asarray([False, False, True, True]),
        "checkpoint_prediction": np.asarray(["X", "Y", "X", "Y"]),
    }

    report, predictions = evaluate(
        arrays,
        descendants={},
        embedding_keys=[],
        knn_device="cpu",
        knn_block_size=2,
        run_knn=False,
    )

    assert report["label_mappings"]["jaccard"] == {"X": "A", "Y": "B"}
    assert predictions["input_row"].tolist() == [2, 3]
    assert predictions["jaccard"].tolist() == ["A", "B"]
