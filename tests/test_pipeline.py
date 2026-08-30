"""Regression tests for the WBCD preprocessing pipeline."""

import numpy as np
from sklearn.datasets import load_breast_cancer


def test_label_encoding_direction():
    """The displayed diagnosis must follow sklearn's non-intuitive encoding."""
    data = load_breast_cancer()

    assert data.target_names[0] == "malignant"
    assert data.target_names[1] == "benign"

    # Import the project's real mapping rather than testing a duplicate mapping.
    from backend.data.pipeline import prediction_to_label

    assert prediction_to_label(0) == "malignant"
    assert prediction_to_label(1) == "benign"


def test_pipeline_shapes_and_real_feature_names():
    from backend.data.pipeline import prepare_breast_cancer_data

    prepared = prepare_breast_cancer_data(random_state=42)

    assert prepared.X_train_full.shape == (455, 30)
    assert prepared.X_test_full.shape == (114, 30)
    assert prepared.X_train_selected.shape == (455, 4)
    assert prepared.X_test_selected.shape == (114, 4)
    assert prepared.X_train_quantum.shape == (455, 4)
    assert prepared.X_test_quantum.shape == (114, 4)
    assert prepared.y.shape == prepared.y_pm1.shape == (569,)
    assert prepared.y_train.shape == prepared.y_train_pm1.shape == (455,)
    assert prepared.y_test.shape == prepared.y_test_pm1.shape == (114,)

    assert prepared.feature_names.shape == (30,)
    assert prepared.selected_feature_names.shape == (4,)
    np.testing.assert_array_equal(
        prepared.selected_feature_names,
        prepared.feature_names[prepared.selector.get_support()],
    )
    assert all(name in prepared.feature_names for name in prepared.selected_feature_names)
    assert not any(name.lower().startswith("pc") for name in prepared.selected_feature_names)


def test_selected_features_match_the_verified_clinical_features():
    from backend.data.pipeline import prepare_breast_cancer_data

    prepared = prepare_breast_cancer_data(random_state=42)

    np.testing.assert_array_equal(
        prepared.selected_feature_names,
        np.array(
            [
                "mean concave points",
                "worst radius",
                "worst perimeter",
                "worst concave points",
            ]
        ),
    )


def test_scalers_are_fit_on_training_data_and_quantum_range_is_bounded():
    from backend.data.pipeline import prepare_breast_cancer_data

    prepared = prepare_breast_cancer_data(random_state=42)
    raw = load_breast_cancer()
    X_train_raw = raw.data[prepared.train_indices]
    X_train_imputed = prepared.imputer.transform(X_train_raw)

    # These fitted values prove the imputer/scaler saw the training partition,
    # rather than being fit before the split and leaking test information.
    np.testing.assert_allclose(
        prepared.imputer.statistics_, np.median(X_train_raw, axis=0)
    )
    np.testing.assert_allclose(prepared.full_scaler.mean_, X_train_imputed.mean(axis=0))
    np.testing.assert_allclose(prepared.X_train_full.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(prepared.X_train_full.std(axis=0), 1.0, atol=1e-12)

    np.testing.assert_allclose(prepared.X_train_quantum.min(axis=0), 0.0)
    np.testing.assert_allclose(prepared.X_train_quantum.max(axis=0), np.pi)
    assert np.all(prepared.X_test_quantum >= 0.0)
    assert np.all(prepared.X_test_quantum <= np.pi)


def test_transform_features_reproduces_all_test_feature_views():
    from backend.data.pipeline import prepare_breast_cancer_data

    prepared = prepare_breast_cancer_data(random_state=42)
    raw = load_breast_cancer()

    full, selected, quantum = prepared.transform_features(
        raw.data[prepared.test_indices]
    )

    np.testing.assert_allclose(full, prepared.X_test_full)
    np.testing.assert_allclose(selected, prepared.X_test_selected)
    np.testing.assert_allclose(quantum, prepared.X_test_quantum)


def test_split_is_stratified_and_labels_remain_separate():
    from backend.data.pipeline import prepare_breast_cancer_data

    prepared = prepare_breast_cancer_data(random_state=42)
    raw = load_breast_cancer()

    assert set(np.unique(prepared.y_train)) == {0, 1}
    assert set(np.unique(prepared.y_test)) == {0, 1}
    assert set(np.unique(prepared.y_train_pm1)) == {-1.0, 1.0}
    assert set(np.unique(prepared.y_test_pm1)) == {-1.0, 1.0}
    np.testing.assert_array_equal(prepared.y, raw.target)
    np.testing.assert_array_equal(prepared.y_pm1, prepared.y * 2 - 1)
    np.testing.assert_array_equal(prepared.y_train_pm1, prepared.y_train * 2 - 1)
    np.testing.assert_array_equal(prepared.y_test_pm1, prepared.y_test * 2 - 1)
    assert not np.shares_memory(prepared.y, prepared.y_pm1)
    assert not np.shares_memory(prepared.y_train, prepared.y_train_pm1)
    assert not np.shares_memory(prepared.y_test, prepared.y_test_pm1)
    np.testing.assert_array_equal(prepared.y_train, raw.target[prepared.train_indices])
    np.testing.assert_array_equal(prepared.y_test, raw.target[prepared.test_indices])

    overall_benign_rate = raw.target.mean()
    assert abs(prepared.y_train.mean() - overall_benign_rate) < 0.01
    assert abs(prepared.y_test.mean() - overall_benign_rate) < 0.01


def test_pipeline_is_deterministic_for_a_given_random_state():
    from backend.data.pipeline import prepare_breast_cancer_data

    first = prepare_breast_cancer_data(random_state=17)
    second = prepare_breast_cancer_data(random_state=17)
    another_seed = prepare_breast_cancer_data(random_state=18)

    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    np.testing.assert_allclose(first.X_train_full, second.X_train_full)
    np.testing.assert_allclose(first.X_train_selected, second.X_train_selected)
    np.testing.assert_allclose(first.X_train_quantum, second.X_train_quantum)
    np.testing.assert_array_equal(
        first.selected_feature_names, second.selected_feature_names
    )
    assert not np.array_equal(first.train_indices, another_seed.train_indices)
