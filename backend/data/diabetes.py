"""Leakage-safe pipeline for the UCI Early Stage Diabetes benchmark.

The public CSV is bundled under CC BY 4.0 so the demo never needs internet
access.  Its questionnaire values are encoded deterministically (Yes/Male=1,
No/Female=0), while the original clinical feature names remain visible in the
API, UI, SHAP, LIME, and reports.

Class ``0`` is the condition-present class (UCI ``Positive``) and class ``1``
is the condition-absent class (UCI ``Negative``).  This mirrors the existing
breast-cancer convention and keeps sensitivity focused on the concerning
condition without ever reusing the breast-specific label mapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from backend.data.pipeline import QUANTUM_RANGE_MAX, QUANTUM_RANGE_MIN


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "early_stage_diabetes.csv"
FEATURE_NAMES = np.asarray(
    [
        "age",
        "gender",
        "polyuria",
        "polydipsia",
        "sudden weight loss",
        "weakness",
        "polyphagia",
        "genital thrush",
        "visual blurring",
        "itching",
        "irritability",
        "delayed healing",
        "partial paresis",
        "muscle stiffness",
        "alopecia",
        "obesity",
    ],
    dtype=str,
)
TARGET_NAMES = np.asarray(
    ["positive screening signal", "negative screening signal"], dtype=str
)


def diabetes_prediction_to_label(prediction: int) -> str:
    """Map the explicit UCI-derived encoding to a cautious screening label."""

    if prediction == 0:
        return str(TARGET_NAMES[0])
    if prediction == 1:
        return str(TARGET_NAMES[1])
    raise ValueError("prediction must be the numeric class 0 or 1")


def load_early_stage_diabetes() -> tuple[FloatArray, IntArray]:
    """Load and encode the unchanged bundled UCI CSV."""

    frame = pd.read_csv(DATASET_PATH)
    expected_columns = [
        "Age",
        "Gender",
        "Polyuria",
        "Polydipsia",
        "sudden weight loss",
        "weakness",
        "Polyphagia",
        "Genital thrush",
        "visual blurring",
        "Itching",
        "Irritability",
        "delayed healing",
        "partial paresis",
        "muscle stiffness",
        "Alopecia",
        "Obesity",
        "class",
    ]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError("bundled UCI diabetes CSV schema does not match its citation")

    features = frame.iloc[:, :-1].copy()
    features["Gender"] = features["Gender"].map({"Female": 0.0, "Male": 1.0})
    for column in expected_columns[2:-1]:
        features[column] = features[column].map({"No": 0.0, "Yes": 1.0})
    targets = frame["class"].map({"Positive": 0, "Negative": 1})
    if features.isna().any().any() or targets.isna().any():
        raise RuntimeError("bundled UCI diabetes CSV contains an unknown categorical value")
    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=np.int64),
    )


@dataclass(slots=True)
class PreparedDiabetesData:
    """Train/test arrays and fitted preprocessing for the diabetes module."""

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


def prepare_diabetes_data(
    *, random_state: int = 42, test_size: float = 0.2
) -> PreparedDiabetesData:
    """Create a deterministic stratified split and train-only transformations."""

    _features, targets = load_early_stage_diabetes()
    indices = np.arange(targets.size, dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )
    return prepare_diabetes_partition(train_indices, test_indices)


def prepare_diabetes_partition(
    train_indices: ArrayLike, test_indices: ArrayLike
) -> PreparedDiabetesData:
    """Fit every preprocessing component on an explicit training partition."""

    features, y = load_early_stage_diabetes()
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if train.ndim != 1 or test.ndim != 1 or train.size == 0 or test.size == 0:
        raise ValueError("both partition index arrays must be non-empty and one-dimensional")
    if np.intersect1d(train, test).size:
        raise ValueError("training and validation indices cannot overlap")
    if len(np.unique(train)) != train.size or len(np.unique(test)) != test.size:
        raise ValueError("partition indices cannot contain duplicates")
    if np.any(train < 0) or np.any(test < 0) or np.any(train >= y.size) or np.any(test >= y.size):
        raise ValueError("partition indices are outside the dataset")
    if set(np.unique(y[train])) != {0, 1} or set(np.unique(y[test])) != {0, 1}:
        raise ValueError("both partitions must contain positive and negative rows")

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(features[train])
    test_imputed = imputer.transform(features[test])
    full_scaler = StandardScaler()
    train_full = full_scaler.fit_transform(train_imputed)
    test_full = full_scaler.transform(test_imputed)
    selector = SelectKBest(score_func=f_classif, k=4)
    train_selected = selector.fit_transform(train_full, y[train])
    test_selected = selector.transform(test_full)
    quantum_scaler = MinMaxScaler(
        feature_range=(QUANTUM_RANGE_MIN, QUANTUM_RANGE_MAX), clip=True
    )
    train_quantum = quantum_scaler.fit_transform(train_selected)
    test_quantum = quantum_scaler.transform(test_selected)
    y_pm1 = (y * 2 - 1).astype(float)

    return PreparedDiabetesData(
        X_train_full=np.asarray(train_full, dtype=float),
        X_test_full=np.asarray(test_full, dtype=float),
        X_train_selected=np.asarray(train_selected, dtype=float),
        X_test_selected=np.asarray(test_selected, dtype=float),
        X_train_quantum=np.asarray(train_quantum, dtype=float),
        X_test_quantum=np.asarray(test_quantum, dtype=float),
        y=y.copy(),
        y_pm1=y_pm1.copy(),
        y_train=y[train].copy(),
        y_test=y[test].copy(),
        y_train_pm1=y_pm1[train].copy(),
        y_test_pm1=y_pm1[test].copy(),
        feature_names=FEATURE_NAMES.copy(),
        selected_feature_names=FEATURE_NAMES[selector.get_support()].copy(),
        target_names=TARGET_NAMES.copy(),
        train_indices=train.copy(),
        test_indices=test.copy(),
        imputer=imputer,
        full_scaler=full_scaler,
        selector=selector,
        quantum_scaler=quantum_scaler,
    )


__all__ = [
    "DATASET_PATH",
    "FEATURE_NAMES",
    "TARGET_NAMES",
    "PreparedDiabetesData",
    "diabetes_prediction_to_label",
    "load_early_stage_diabetes",
    "prepare_diabetes_data",
    "prepare_diabetes_partition",
]
