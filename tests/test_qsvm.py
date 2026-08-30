"""Regression tests for the two Phase 2 quantum-kernel SVM variants."""

from __future__ import annotations

import numpy as np
import pytest

from backend.quantum.qsvm_kernels import QuantumKernelSVM


ANGLE_KERNEL_ROWS = np.asarray(
    [
        [0.10, 0.30, 0.70, 1.20],
        [0.50, 0.20, 1.10, 0.40],
        [1.20, 0.90, 0.30, 0.20],
        [0.80, 1.30, 0.60, 0.90],
    ],
    dtype=float,
)


def test_angle_kernel_matrix_is_symmetric_psd_with_unit_diagonal():
    qsvm = QuantumKernelSVM(embedding="angle")

    matrix = qsvm.square_kernel_matrix(ANGLE_KERNEL_ROWS)

    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(matrix), np.ones(4), atol=1e-10)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-9


def test_amplitude_embedding_uses_exactly_two_qubits_for_four_features():
    qsvm = QuantumKernelSVM(embedding="amplitude")

    assert qsvm.n_qubits == 2
    assert qsvm.n_features == 4
    assert qsvm.kernel(ANGLE_KERNEL_ROWS[0], ANGLE_KERNEL_ROWS[1]) >= 0.0

    with pytest.raises(ValueError, match="exactly 2 qubits"):
        QuantumKernelSVM(embedding="amplitude", n_qubits=4)

    with pytest.raises(ValueError, match="exactly 4 features"):
        qsvm.kernel(np.ones(16), np.ones(16))

    # MinMax scaling can produce an exact all-zero row. AmplitudeEmbedding
    # cannot normalize it, so the wrapper uses the deterministic |00> state.
    zero_kernel = qsvm.square_kernel_matrix(
        np.asarray([np.zeros(4), ANGLE_KERNEL_ROWS[0]])
    )
    assert np.all(np.isfinite(zero_kernel))
    np.testing.assert_allclose(np.diag(zero_kernel), np.ones(2), atol=1e-10)


def test_amplitude_zero_vector_fallback_is_general_and_does_not_mutate_inference_input():
    X_train = np.asarray(
        [
            [0.10, 0.15, 0.20, 0.25],
            [0.20, 0.25, 0.30, 0.35],
            [0.30, 0.35, 0.40, 0.45],
            [2.30, 2.35, 2.40, 2.45],
            [2.50, 2.55, 2.60, 2.65],
            [2.70, 2.75, 2.80, 2.85],
        ],
        dtype=float,
    )
    y_train = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    qsvm = QuantumKernelSVM(embedding="amplitude", random_state=7).fit(
        X_train, y_train
    )
    live_like_zeros = np.zeros((3, 4), dtype=float)
    original = live_like_zeros.copy()

    probabilities = qsvm.predict_proba(live_like_zeros)

    assert probabilities.shape == (3, 2)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3), atol=1e-10)
    np.testing.assert_array_equal(live_like_zeros, original)


@pytest.fixture(scope="module")
def fitted_angle_qsvm() -> QuantumKernelSVM:
    # Both classes are deliberately present; 1 means benign in sklearn's WBCD
    # encoding, while 0 means malignant.
    X_train = np.asarray(
        [
            [0.10, 0.15, 0.20, 0.25],
            [0.20, 0.25, 0.30, 0.35],
            [0.30, 0.35, 0.40, 0.45],
            [2.30, 2.35, 2.40, 2.45],
            [2.50, 2.55, 2.60, 2.65],
            [2.70, 2.75, 2.80, 2.85],
        ],
        dtype=float,
    )
    y_train = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    return QuantumKernelSVM(embedding="angle", random_state=7).fit(
        X_train, y_train
    )


def test_qsvm_fits_svc_with_precomputed_kernel_and_predicts_probabilities(
    fitted_angle_qsvm: QuantumKernelSVM,
):
    X_test = np.asarray(
        [[0.18, 0.23, 0.28, 0.33], [2.55, 2.60, 2.65, 2.70]],
        dtype=float,
    )

    predictions = fitted_angle_qsvm.predict(X_test)
    probabilities = fitted_angle_qsvm.predict_proba(X_test)

    assert fitted_angle_qsvm.model_.kernel == "precomputed"
    assert fitted_angle_qsvm.train_kernel_matrix_.shape == (6, 6)
    assert predictions.shape == (2,)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-10)


def test_benign_probability_selects_the_class_one_column(
    fitted_angle_qsvm: QuantumKernelSVM,
):
    X_test = np.asarray(
        [[0.18, 0.23, 0.28, 0.33], [2.55, 2.60, 2.65, 2.70]],
        dtype=float,
    )

    probabilities = fitted_angle_qsvm.predict_proba(X_test)
    benign_column = int(np.flatnonzero(fitted_angle_qsvm.classes_ == 1)[0])

    np.testing.assert_allclose(
        fitted_angle_qsvm.predict_benign_proba(X_test),
        probabilities[:, benign_column],
    )
