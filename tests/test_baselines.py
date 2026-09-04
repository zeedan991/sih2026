"""Tests for both required classical-baseline configurations."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.classical.baselines import (
    FEATURE_CONFIGURATIONS,
    MODEL_NAMES,
    POSITIVE_CLASS_NAME,
    POSITIVE_LABEL,
    evaluate_classification,
    train_evaluate_baselines,
)
from backend.data.pipeline import prepare_breast_cancer_data


@pytest.fixture(scope="module")
def prepared_data():
    return prepare_breast_cancer_data(random_state=42)


@pytest.fixture(scope="module")
def baseline_results(prepared_data):
    return train_evaluate_baselines(prepared_data, random_state=42)


def test_all_eight_model_configuration_combinations_are_reported(
    prepared_data, baseline_results
) -> None:
    assert tuple(baseline_results) == FEATURE_CONFIGURATIONS

    for configuration in FEATURE_CONFIGURATIONS:
        assert tuple(baseline_results[configuration]) == MODEL_NAMES
        for model_name in MODEL_NAMES:
            result = baseline_results[configuration][model_name]
            assert result.model_name == model_name
            assert result.feature_configuration == configuration
            assert result.predictions.shape == prepared_data.y_test.shape
            assert set(np.unique(result.predictions)) <= {0, 1}


def test_configurations_use_30_and_exact_same_4_selected_features(
    prepared_data, baseline_results
) -> None:
    assert prepared_data.X_train_full.shape[1] == 30
    assert prepared_data.X_train_selected.shape[1] == 4
    np.testing.assert_allclose(
        prepared_data.X_train_selected,
        prepared_data.selector.transform(prepared_data.X_train_full),
    )

    for result in baseline_results["full_feature"].values():
        assert result.estimator.n_features_in_ == 30
    for result in baseline_results["same_4_feature"].values():
        assert result.estimator.n_features_in_ == 4


def test_metrics_use_malignant_as_the_clinical_positive_class(
    prepared_data, baseline_results
) -> None:
    assert POSITIVE_LABEL == 0
    assert POSITIVE_CLASS_NAME == "malignant"
    assert prepared_data.target_names[POSITIVE_LABEL] == POSITIVE_CLASS_NAME

    for configuration in FEATURE_CONFIGURATIONS:
        for result in baseline_results[configuration].values():
            predictions = result.predictions
            metrics = result.metrics
            assert metrics.positive_label == 0
            assert metrics.positive_class_name == "malignant"
            assert metrics.accuracy == pytest.approx(
                accuracy_score(prepared_data.y_test, predictions)
            )
            assert metrics.precision == pytest.approx(
                precision_score(prepared_data.y_test, predictions, pos_label=0)
            )
            assert metrics.recall == pytest.approx(
                recall_score(prepared_data.y_test, predictions, pos_label=0)
            )
            assert metrics.f1 == pytest.approx(
                f1_score(prepared_data.y_test, predictions, pos_label=0)
            )
            assert metrics.malignant_sensitivity == pytest.approx(metrics.recall)
            assert metrics.specificity == pytest.approx(
                recall_score(prepared_data.y_test, predictions, pos_label=1)
            )
            expected = confusion_matrix(
                prepared_data.y_test, predictions, labels=[0, 1]
            )
            assert metrics.confusion_matrix == {
                "true_positive": int(expected[0, 0]),
                "false_negative": int(expected[0, 1]),
                "false_positive": int(expected[1, 0]),
                "true_negative": int(expected[1, 1]),
            }
            assert metrics.roc_auc is not None
            assert 0.0 <= metrics.roc_auc <= 1.0


def test_training_preserves_original_classical_labels(
    prepared_data, baseline_results
) -> None:
    np.testing.assert_array_equal(
        prepared_data.y_train_pm1,
        prepared_data.y_train * 2 - 1,
    )
    assert set(np.unique(prepared_data.y_train)) == {0, 1}
    assert set(np.unique(prepared_data.y_test)) == {0, 1}

    for configuration in FEATURE_CONFIGURATIONS:
        for result in baseline_results[configuration].values():
            np.testing.assert_array_equal(result.estimator.classes_, np.array([0, 1]))


def test_baselines_are_deterministic_for_fixed_seed(
    prepared_data, baseline_results
) -> None:
    repeated = train_evaluate_baselines(prepared_data, random_state=42)

    for configuration in FEATURE_CONFIGURATIONS:
        for model_name in MODEL_NAMES:
            first = baseline_results[configuration][model_name]
            second = repeated[configuration][model_name]
            np.testing.assert_array_equal(first.predictions, second.predictions)
            assert first.metrics == second.metrics


def test_metric_validation_rejects_pm1_labels() -> None:
    with pytest.raises(ValueError, match="original labels"):
        evaluate_classification(
            np.array([-1, 1], dtype=int),
            np.array([0, 1], dtype=int),
        )
