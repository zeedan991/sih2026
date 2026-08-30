"""Fast regression tests for Phase 2 ensemble mathematics."""

from __future__ import annotations

import numpy as np
import pytest

from backend.quantum.ensemble import (
    OOBPrediction,
    inverse_oob_mse_weights,
    make_oob_bootstrap,
    paired_correctness_ttest,
    weighted_soft_vote,
)


def test_oob_bootstrap_is_reproducible_and_really_out_of_bag() -> None:
    first = make_oob_bootstrap(n_samples=40, seed=123)
    second = make_oob_bootstrap(n_samples=40, seed=123)

    np.testing.assert_array_equal(first.bootstrap_indices, second.bootstrap_indices)
    np.testing.assert_array_equal(first.oob_indices, second.oob_indices)
    assert first.bootstrap_indices.shape == (40,)
    assert first.oob_indices.size > 0
    assert set(first.oob_indices).isdisjoint(set(first.bootstrap_indices))


def test_inverse_oob_mse_weights_favour_the_better_calibrated_model() -> None:
    targets = np.array([0, 0, 1, 1], dtype=int)
    predictions = {
        "good": OOBPrediction(targets, np.array([0.05, 0.1, 0.9, 0.95])),
        "weak": OOBPrediction(targets, np.array([0.4, 0.45, 0.55, 0.6])),
    }

    result = inverse_oob_mse_weights(predictions)

    assert result.weights["good"] > result.weights["weak"]
    assert result.mse_by_model["good"] < result.mse_by_model["weak"]
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_weighted_soft_vote_uses_probabilities_not_hard_labels() -> None:
    probabilities = {
        "vqc": np.array([0.2, 0.7, 0.4]),
        "qsvm": np.array([0.8, 0.6, 0.9]),
    }
    weights = {"vqc": 0.25, "qsvm": 0.75}

    voted = weighted_soft_vote(probabilities, weights)

    np.testing.assert_allclose(voted, np.array([0.65, 0.625, 0.775]))


def test_weighted_soft_vote_requires_matching_model_names() -> None:
    with pytest.raises(ValueError, match="same model names"):
        weighted_soft_vote(
            {"vqc": np.array([0.5])},
            {"different": 1.0},
        )


def test_paired_correctness_ttest_reports_ensemble_improvement() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    best_single = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    ensemble = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    comparison = paired_correctness_ttest(y_true, ensemble, best_single)

    assert comparison.ensemble_accuracy == pytest.approx(1.0)
    assert comparison.single_accuracy == pytest.approx(4 / 6)
    assert comparison.mean_correctness_difference == pytest.approx(2 / 6)
    assert comparison.statistic > 0
    assert 0.0 <= comparison.p_value <= 1.0


def test_paired_correctness_ttest_handles_identical_predictions() -> None:
    y_true = np.array([0, 1, 1, 0], dtype=int)
    predictions = np.array([0, 1, 0, 0], dtype=int)

    comparison = paired_correctness_ttest(y_true, predictions, predictions)

    assert comparison.statistic == 0.0
    assert comparison.p_value == 1.0
    assert comparison.mean_correctness_difference == 0.0
