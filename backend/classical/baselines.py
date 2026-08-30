"""Deterministic classical baselines for both required feature views.

The full-feature configuration is the realistic classical benchmark and uses
all 30 standardized WBCD features.  The same-4-feature configuration uses the
exact four standardized columns selected for the quantum path, before the
additional ``[0, pi]`` quantum-only scaling.  Reporting both is required by
decision D-14.

All classification metrics in this module use sklearn's original target
encoding with class ``1`` (benign) as the positive class.  Class ``0`` is
malignant; the labels are never converted to ``{-1, +1}`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

from backend.data.pipeline import PreparedBreastCancerData


IntArray: TypeAlias = NDArray[np.int64]
FeatureConfiguration: TypeAlias = Literal["full_feature", "same_4_feature"]

MODEL_NAMES: Final[tuple[str, ...]] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "svm",
)
FEATURE_CONFIGURATIONS: Final[tuple[FeatureConfiguration, ...]] = (
    "full_feature",
    "same_4_feature",
)
POSITIVE_LABEL: Final[int] = 1
POSITIVE_CLASS_NAME: Final[str] = "benign"


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Held-out metrics with benign (sklearn label 1) as positive."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_label: int = POSITIVE_LABEL
    positive_class_name: str = POSITIVE_CLASS_NAME


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One fitted model/configuration result on the held-out test set."""

    model_name: str
    feature_configuration: FeatureConfiguration
    estimator: ClassifierMixin
    predictions: IntArray
    metrics: ClassificationMetrics


BaselineResults: TypeAlias = dict[
    FeatureConfiguration, dict[str, BaselineResult]
]


def build_baseline_models(*, random_state: int) -> dict[str, ClassifierMixin]:
    """Create the four fixed, seeded Phase 1 baseline estimators."""

    return {
        "logistic_regression": LogisticRegression(
            solver="lbfgs",
            max_iter=2_000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            # Phase 1 only needs labels/metrics.  Omitting sklearn 1.9's
            # deprecated ``probability`` argument also avoids D-18's warning.
            random_state=random_state,
        ),
    }


def evaluate_classification(
    y_true: NDArray[np.integer], predictions: NDArray[np.integer]
) -> ClassificationMetrics:
    """Evaluate original ``{0,1}`` labels with benign as positive."""

    y_true_array = np.asarray(y_true, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    if y_true_array.ndim != 1 or prediction_array.ndim != 1:
        raise ValueError("y_true and predictions must be one-dimensional")
    if y_true_array.shape != prediction_array.shape:
        raise ValueError("y_true and predictions must have matching shapes")
    if not np.isin(y_true_array, (0, 1)).all():
        raise ValueError("y_true must use original labels {0,1}")
    if not np.isin(prediction_array, (0, 1)).all():
        raise ValueError("predictions must use original labels {0,1}")

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true_array, prediction_array)),
        precision=float(
            precision_score(
                y_true_array,
                prediction_array,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true_array,
                prediction_array,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true_array,
                prediction_array,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
    )


def train_evaluate_baselines(
    data: PreparedBreastCancerData,
    *,
    random_state: int,
) -> BaselineResults:
    """Fit and evaluate all eight model/configuration combinations.

    Both configurations use the same train/test row indices and the same
    original ``y_train``/``y_test`` arrays.  The only difference is whether a
    model receives all 30 standardized features or the selected four.
    """

    feature_views: dict[
        FeatureConfiguration, tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = {
        "full_feature": (data.X_train_full, data.X_test_full),
        "same_4_feature": (data.X_train_selected, data.X_test_selected),
    }
    results: BaselineResults = {}

    for configuration, (X_train, X_test) in feature_views.items():
        configuration_results: dict[str, BaselineResult] = {}
        for model_name, estimator in build_baseline_models(
            random_state=random_state
        ).items():
            estimator.fit(X_train, data.y_train)
            predictions = np.asarray(estimator.predict(X_test), dtype=np.int64)
            configuration_results[model_name] = BaselineResult(
                model_name=model_name,
                feature_configuration=configuration,
                estimator=estimator,
                predictions=predictions,
                metrics=evaluate_classification(data.y_test, predictions),
            )
        results[configuration] = configuration_results

    return results
