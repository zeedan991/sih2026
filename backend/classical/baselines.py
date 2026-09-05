"""Deterministic classical baselines for both required feature views.

The full-feature configuration is the realistic classical benchmark and uses
all standardized dataset features.  The same-4-feature configuration uses the
exact four standardized columns selected for the quantum path, before the
additional inward ``(0, pi)`` quantum-only scaling. Reporting both is required by
decision D-14.

Clinical metrics explicitly treat class ``0`` (condition present) as positive.
For the original WBCD module that is malignant; each additional disease module
supplies its own clinically accurate class name.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
POSITIVE_LABEL: Final[int] = 0
POSITIVE_CLASS_NAME: Final[str] = "malignant"


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Disease-oriented metrics with class 0 as the concerning condition."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    malignant_sensitivity: float
    condition_sensitivity: float
    specificity: float
    confusion_matrix: dict[str, int]
    roc_auc: float | None
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
    fit_seconds: float
    predict_seconds: float


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
    y_true: NDArray[np.integer],
    predictions: NDArray[np.integer],
    *,
    malignant_scores: NDArray[np.floating] | None = None,
    positive_class_name: str = POSITIVE_CLASS_NAME,
) -> ClassificationMetrics:
    """Evaluate binary labels with class 0 as the clinical positive class.

    ``malignant_scores`` retains its historic keyword for backwards
    compatibility; for non-oncology modules it means the class-zero condition
    score.
    """

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

    matrix = confusion_matrix(y_true_array, prediction_array, labels=[0, 1])
    true_positive, false_negative = (int(value) for value in matrix[0])
    false_positive, true_negative = (int(value) for value in matrix[1])
    sensitivity = float(
        recall_score(y_true_array, prediction_array, pos_label=0, zero_division=0)
    )
    roc_auc: float | None = None
    if malignant_scores is not None:
        score_array = np.asarray(malignant_scores, dtype=float)
        if score_array.shape != y_true_array.shape or not np.isfinite(score_array).all():
            raise ValueError("malignant_scores must be finite and match y_true")
        roc_auc = float(roc_auc_score(y_true_array == 0, score_array))

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true_array, prediction_array)),
        precision=float(
            precision_score(
                y_true_array,
                prediction_array,
                pos_label=0,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true_array,
                prediction_array,
                pos_label=0,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true_array,
                prediction_array,
                pos_label=0,
                zero_division=0,
            )
        ),
        malignant_sensitivity=sensitivity,
        condition_sensitivity=sensitivity,
        specificity=float(
            recall_score(y_true_array, prediction_array, pos_label=1, zero_division=0)
        ),
        confusion_matrix={
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "true_negative": true_negative,
        },
        roc_auc=roc_auc,
        positive_class_name=positive_class_name,
    )


def malignant_score(
    estimator: ClassifierMixin,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a continuous score where larger means class-zero/condition present."""

    classes = np.asarray(estimator.classes_, dtype=np.int64)
    matches = np.flatnonzero(classes == 0)
    if matches.size != 1:
        raise RuntimeError("estimator must contain condition-present class 0")
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
        return probabilities[:, int(matches[0])]
    decisions = np.asarray(estimator.decision_function(features), dtype=float)
    if decisions.ndim == 1:
        return -decisions if tuple(classes) == (0, 1) else decisions
    return decisions[:, int(matches[0])]


def train_evaluate_baselines(
    data: PreparedBreastCancerData,
    *,
    random_state: int,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> BaselineResults:
    """Fit and evaluate all eight model/configuration combinations.

    Both configurations use the same train/test row indices and the same
    original ``y_train``/``y_test`` arrays.  The only difference is whether a
    model receives all 30 standardized features or the selected four.
    """

    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("model_names must contain unique baseline model names")
    unknown = set(model_names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown baseline model names: {sorted(unknown)}")

    feature_views: dict[
        FeatureConfiguration, tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = {
        "full_feature": (data.X_train_full, data.X_test_full),
        "same_4_feature": (data.X_train_selected, data.X_test_selected),
    }
    results: BaselineResults = {}

    for configuration, (X_train, X_test) in feature_views.items():
        configuration_results: dict[str, BaselineResult] = {}
        available_models = build_baseline_models(random_state=random_state)
        for model_name in model_names:
            estimator = available_models[model_name]
            fit_started = time.perf_counter()
            estimator.fit(X_train, data.y_train)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            predictions = np.asarray(estimator.predict(X_test), dtype=np.int64)
            scores = malignant_score(estimator, X_test)
            predict_seconds = time.perf_counter() - predict_started
            configuration_results[model_name] = BaselineResult(
                model_name=model_name,
                feature_configuration=configuration,
                estimator=estimator,
                predictions=predictions,
                metrics=evaluate_classification(
                    data.y_test,
                    predictions,
                    malignant_scores=scores,
                    positive_class_name=str(getattr(data, "target_names", [POSITIVE_CLASS_NAME])[0]),
                ),
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
            )
        results[configuration] = configuration_results

    return results
