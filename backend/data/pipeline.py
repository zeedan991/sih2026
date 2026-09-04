"""Leakage-safe preprocessing for the Wisconsin Breast Cancer dataset.

The returned data keeps three explicit feature views:

* all 30 imputed and standardized features for realistic baselines;
* the same four named selected features for fair classical comparisons;
* those four features additionally scaled just inside ``(0, pi)`` for quantum
  models, avoiding exact rotation/amplitude boundaries during live inference.

Classical and quantum labels are separate arrays throughout.  The original
sklearn encoding is never mutated: 0 is malignant and 1 is benign.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

# Keep quantum inputs strictly away from exact range boundaries.  ``clip=True``
# then gives held-out and future API rows the same guarantee as training rows.
QUANTUM_RANGE_EPSILON = 1.0e-6
QUANTUM_RANGE_MIN = QUANTUM_RANGE_EPSILON
QUANTUM_RANGE_MAX = float(np.pi) - QUANTUM_RANGE_EPSILON


def prediction_to_label(prediction: int) -> str:
    """Map a sklearn class prediction to the correct clinical label.

    This intentionally accepts class predictions only, rather than silently
    treating arbitrary probabilities as classes.  Quantum probabilities must
    first be thresholded into the same ``{0, 1}`` encoding.
    """

    if isinstance(prediction, bool) or not isinstance(prediction, (Integral, Real)):
        raise ValueError("prediction must be the numeric class 0 or 1")
    if prediction == 0:
        return "malignant"
    if prediction == 1:
        return "benign"
    raise ValueError("prediction must be the numeric class 0 or 1")


@dataclass(slots=True)
class PreparedBreastCancerData:
    """Train/test data and fitted preprocessing components for downstream models."""

    X_train_full: FloatArray
    X_test_full: FloatArray
    X_train_selected: FloatArray
    X_test_selected: FloatArray
    X_train_quantum: FloatArray
    X_test_quantum: FloatArray
    y: IntArray
    y_pm1: FloatArray
    y_train: IntArray
    y_test: IntArray
    y_train_pm1: FloatArray
    y_test_pm1: FloatArray
    feature_names: StringArray
    selected_feature_names: StringArray
    target_names: StringArray
    train_indices: IntArray
    test_indices: IntArray
    imputer: SimpleImputer
    full_scaler: StandardScaler
    selector: SelectKBest
    quantum_scaler: MinMaxScaler

    def transform_features(
        self, features: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Transform raw 30-feature rows into all three fitted feature views.

        The quantum view is clipped strictly inside ``(0, pi)`` by the fitted
        scaler, including for raw live-inference values outside its fit range.
        """

        array = np.asarray(features, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.feature_names.size:
            raise ValueError(
                f"features must have shape (n_samples, {self.feature_names.size})"
            )

        imputed = self.imputer.transform(array)
        full = self.full_scaler.transform(imputed)
        selected = self.selector.transform(full)
        quantum = self.quantum_scaler.transform(selected)
        return full, selected, quantum


def prepare_breast_cancer_data(
    *, random_state: int = 42, test_size: float = 0.2
) -> PreparedBreastCancerData:
    """Load WBCD and create deterministic, stratified, leakage-safe views.

    The raw rows are split before any preprocessing component is fitted.  The
    median imputer, standard scaler, ANOVA selector, and quantum-range scaler
    therefore learn exclusively from training data.
    """

    dataset = load_breast_cancer()
    y = np.asarray(dataset.target, dtype=np.int64)
    indices = np.arange(y.size, dtype=np.int64)

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return prepare_breast_cancer_partition(train_indices, test_indices)


def prepare_breast_cancer_partition(
    train_indices: ArrayLike,
    test_indices: ArrayLike,
) -> PreparedBreastCancerData:
    """Fit the complete pipeline on an explicit train/validation partition.

    This is the fold-safe entry point used by cross-validation.  Every fitted
    preprocessing component sees only ``train_indices``; validation rows are
    transformed afterward with no refitting.
    """

    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=np.int64).copy()
    y_pm1 = (y * 2 - 1).astype(float)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    if train_indices.ndim != 1 or test_indices.ndim != 1:
        raise ValueError("partition indices must be one-dimensional")
    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("both partition index arrays must be non-empty")
    if (
        len(np.unique(train_indices)) != train_indices.size
        or len(np.unique(test_indices)) != test_indices.size
    ):
        raise ValueError("partition indices cannot contain duplicates")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("training and validation indices cannot overlap")
    if (
        np.any(train_indices < 0)
        or np.any(test_indices < 0)
        or np.any(train_indices >= y.size)
        or np.any(test_indices >= y.size)
    ):
        raise ValueError("partition indices are outside the dataset")
    if (
        set(np.unique(y[train_indices])) != {0, 1}
        or set(np.unique(y[test_indices])) != {0, 1}
    ):
        raise ValueError("both partitions must contain malignant and benign rows")

    X_train_raw = X[train_indices]
    X_test_raw = X[test_indices]
    y_train = y[train_indices].copy()
    y_test = y[test_indices].copy()

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    full_scaler = StandardScaler()
    X_train_full = full_scaler.fit_transform(X_train_imputed)
    X_test_full = full_scaler.transform(X_test_imputed)

    selector = SelectKBest(score_func=f_classif, k=4)
    X_train_selected = selector.fit_transform(X_train_full, y_train)
    X_test_selected = selector.transform(X_test_full)

    # The inward range avoids exact boundary vectors. clip=True applies the
    # same guarantee to unseen test rows and future live/API predictions.
    quantum_scaler = MinMaxScaler(
        feature_range=(QUANTUM_RANGE_MIN, QUANTUM_RANGE_MAX), clip=True
    )
    X_train_quantum = quantum_scaler.fit_transform(X_train_selected)
    X_test_quantum = quantum_scaler.transform(X_test_selected)

    # Create distinct float arrays; never mutate or alias the classical labels.
    y_train_pm1 = y_pm1[train_indices].copy()
    y_test_pm1 = y_pm1[test_indices].copy()

    feature_names = np.asarray(dataset.feature_names, dtype=str).copy()
    selected_feature_names = feature_names[selector.get_support()].copy()
    target_names = np.asarray(dataset.target_names, dtype=str).copy()

    return PreparedBreastCancerData(
        X_train_full=np.asarray(X_train_full, dtype=float),
        X_test_full=np.asarray(X_test_full, dtype=float),
        X_train_selected=np.asarray(X_train_selected, dtype=float),
        X_test_selected=np.asarray(X_test_selected, dtype=float),
        X_train_quantum=np.asarray(X_train_quantum, dtype=float),
        X_test_quantum=np.asarray(X_test_quantum, dtype=float),
        y=y,
        y_pm1=y_pm1,
        y_train=y_train,
        y_test=y_test,
        y_train_pm1=y_train_pm1,
        y_test_pm1=y_test_pm1,
        feature_names=feature_names,
        selected_feature_names=selected_feature_names,
        target_names=target_names,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        imputer=imputer,
        full_scaler=full_scaler,
        selector=selector,
        quantum_scaler=quantum_scaler,
    )
