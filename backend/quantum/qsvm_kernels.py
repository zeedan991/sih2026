"""Fidelity quantum kernels and precomputed-kernel SVM wrappers.

The two variants deliberately share the project's four selected clinical
features but not the same device size.  Angle embedding uses four qubits,
while amplitude embedding uses exactly two: four amplitudes fill a two-qubit
state because ``2**2 == 4``.  Expanding the latter to four qubits would require
16 input values and is rejected explicitly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pennylane as qml
from numpy.typing import ArrayLike, NDArray
from sklearn.svm import SVC


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Embedding = Literal["angle", "amplitude"]


class QuantumKernelSVM:
    """Binary QSVM backed by a PennyLane fidelity kernel.

    The fitted scikit-learn estimator receives quantum kernel matrices through
    ``SVC(kernel="precomputed", probability=True)``.  The resulting class-one
    probability is the benign probability because Wisconsin Breast Cancer
    labels are ``0=malignant`` and ``1=benign``.

    The ``probability=True`` argument raises an expected ``FutureWarning`` on
    the repository's pinned scikit-learn 1.9.0 (decision D-18).  It remains the
    specified, working API for this pinned environment.
    """

    ANGLE_QUBITS = 4
    AMPLITUDE_QUBITS = 2
    N_FEATURES = 4

    def __init__(
        self,
        *,
        embedding: Embedding,
        n_qubits: int | None = None,
        C: float = 1.0,
        random_state: int | None = 42,
        device_name: str = "lightning.qubit",
    ) -> None:
        if embedding not in ("angle", "amplitude"):
            raise ValueError("embedding must be 'angle' or 'amplitude'")

        expected_qubits = (
            self.ANGLE_QUBITS
            if embedding == "angle"
            else self.AMPLITUDE_QUBITS
        )
        actual_qubits = expected_qubits if n_qubits is None else n_qubits
        if actual_qubits != expected_qubits:
            if embedding == "amplitude":
                raise ValueError(
                    "amplitude embedding requires exactly 2 qubits for "
                    "the project's 4 selected features"
                )
            raise ValueError(
                "angle embedding requires exactly 4 qubits for the project's "
                "4 selected features"
            )
        if C <= 0.0:
            raise ValueError("C must be positive")

        self.embedding = embedding
        self.n_qubits = actual_qubits
        self.n_features = self.N_FEATURES
        self.C = float(C)
        self.random_state = random_state
        self.device_name = device_name
        self._wires = tuple(range(self.n_qubits))
        self._device = qml.device(device_name, wires=self.n_qubits)
        self._kernel_circuit = self._build_kernel_circuit()

    def _build_kernel_circuit(self) -> qml.QNode:
        wires = self._wires
        device = self._device

        if self.embedding == "angle":

            @qml.qnode(device)
            def kernel_circuit(x1: FloatArray, x2: FloatArray):
                qml.AngleEmbedding(x1, wires=wires)
                qml.adjoint(qml.AngleEmbedding)(x2, wires=wires)
                return qml.expval(qml.Projector([0] * self.n_qubits, wires=wires))

        else:

            @qml.qnode(device)
            def kernel_circuit(x1: FloatArray, x2: FloatArray):
                qml.AmplitudeEmbedding(
                    x1, wires=wires, normalize=True, pad_with=0.0
                )
                qml.adjoint(qml.AmplitudeEmbedding)(
                    x2, wires=wires, normalize=True, pad_with=0.0
                )
                return qml.expval(qml.Projector([0] * self.n_qubits, wires=wires))

        return kernel_circuit

    def _feature_vector(self, values: ArrayLike, *, name: str) -> FloatArray:
        vector = np.asarray(values, dtype=float)
        if vector.ndim != 1 or vector.size != self.n_features:
            raise ValueError(f"{name} must contain exactly 4 features")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        vector = np.asarray(vector, dtype=float).copy()
        if self.embedding == "amplitude" and np.linalg.norm(vector) == 0.0:
            # A row containing the training minima for all four selected
            # features becomes exactly zero after MinMax scaling.  A zero
            # vector has no amplitude direction and PennyLane would divide by
            # zero while normalizing it, so choose the canonical |00> state.
            vector[0] = 1.0
        return vector

    def _feature_matrix(self, values: ArrayLike, *, name: str) -> FloatArray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features:
            raise ValueError(f"{name} must have shape (n_samples, 4)")
        if matrix.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values")
        matrix = np.asarray(matrix, dtype=float).copy()
        if self.embedding == "amplitude":
            zero_rows = np.linalg.norm(matrix, axis=1) == 0.0
            matrix[zero_rows, 0] = 1.0
        return matrix

    def kernel(self, x1: ArrayLike, x2: ArrayLike) -> float:
        """Return the all-zero-projector fidelity between two feature rows."""

        first = self._feature_vector(x1, name="x1")
        second = self._feature_vector(x2, name="x2")
        value = float(qml.math.toarray(self._kernel_circuit(first, second)))
        # Round-off can put a theoretical probability a few ulps outside [0, 1].
        return float(np.clip(value, 0.0, 1.0))

    def square_kernel_matrix(self, features: ArrayLike) -> FloatArray:
        """Build a symmetric train kernel matrix with known unit diagonal."""

        X = self._feature_matrix(features, name="features")
        matrix = qml.kernels.square_kernel_matrix(
            X, self.kernel, assume_normalized_kernel=True
        )
        return np.asarray(qml.math.toarray(matrix), dtype=float)

    def kernel_matrix(self, left: ArrayLike, right: ArrayLike) -> FloatArray:
        """Build a rectangular kernel matrix for two feature collections."""

        X_left = self._feature_matrix(left, name="left")
        X_right = self._feature_matrix(right, name="right")
        matrix = qml.kernels.kernel_matrix(X_left, X_right, self.kernel)
        return np.asarray(qml.math.toarray(matrix), dtype=float)

    def fit(self, features: ArrayLike, labels: ArrayLike) -> QuantumKernelSVM:
        """Fit an SVC using a cached quantum train kernel matrix."""

        X = self._feature_matrix(features, name="features")
        y = np.asarray(labels)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("labels must have shape (n_samples,)")
        if not np.all(np.isin(y, (0, 1))) or set(np.unique(y)) != {0, 1}:
            raise ValueError("labels must contain both WBCD classes 0 and 1")

        self.train_features_ = X.copy()
        self.train_kernel_matrix_ = self.square_kernel_matrix(X)
        self.model_ = SVC(
            C=self.C,
            kernel="precomputed",
            probability=True,
            random_state=self.random_state,
        )
        self.model_.fit(self.train_kernel_matrix_, y.astype(np.int64, copy=False))
        return self

    def _test_kernel_matrix(self, features: ArrayLike) -> FloatArray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before prediction")
        return self.kernel_matrix(features, self.train_features_)

    @property
    def classes_(self) -> IntArray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before classes_ is available")
        return np.asarray(self.model_.classes_, dtype=np.int64)

    def predict(self, features: ArrayLike) -> IntArray:
        """Predict original WBCD labels: 0=malignant, 1=benign."""

        matrix = self._test_kernel_matrix(features)
        return np.asarray(self.model_.predict(matrix), dtype=np.int64)

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return SVC probabilities in ``classes_`` column order."""

        matrix = self._test_kernel_matrix(features)
        return np.asarray(self.model_.predict_proba(matrix), dtype=float)

    def predict_benign_proba(self, features: ArrayLike) -> FloatArray:
        """Return the probability of class 1 (benign), never a guessed column."""

        probabilities = self.predict_proba(features)
        matches = np.flatnonzero(self.classes_ == 1)
        if matches.size != 1:
            raise RuntimeError("fitted QSVM does not contain benign class 1")
        return probabilities[:, int(matches[0])]

    def score(self, features: ArrayLike, labels: ArrayLike) -> float:
        """Return classification accuracy using the cached training features."""

        y = np.asarray(labels)
        return float(np.mean(self.predict(features) == y))
