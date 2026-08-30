"""Out-of-bag weighting and weighted soft voting for quantum models.

The helpers in this module are deliberately model-agnostic. Each VQC and
QSVM is trained on its own bootstrap sample. Its probability calibration is
then scored only on rows absent from that bootstrap (the out-of-bag rows), so
the held-out test set never influences ensemble weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ttest_rel


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_MSE_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class OOBBootstrap:
    """Indices for one bootstrap training sample and its held-out OOB rows."""

    bootstrap_indices: IntArray
    oob_indices: IntArray


@dataclass(frozen=True, slots=True)
class OOBPrediction:
    """Original ``{0,1}`` labels and positive-class probabilities on OOB rows."""

    targets: IntArray
    positive_probabilities: FloatArray


@dataclass(frozen=True, slots=True)
class OOBWeightingResult:
    """Normalized inverse-MSE weights plus their auditable source errors."""

    weights: dict[str, float]
    mse_by_model: dict[str, float]


@dataclass(frozen=True, slots=True)
class PairedCorrectnessComparison:
    """Paired t-test over per-patient correctness indicators."""

    statistic: float
    p_value: float
    ensemble_accuracy: float
    single_accuracy: float
    mean_correctness_difference: float


def make_oob_bootstrap(*, n_samples: int, seed: int) -> OOBBootstrap:
    """Draw a deterministic size-``n`` bootstrap with at least one OOB row."""

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 for OOB estimation")

    rng = np.random.default_rng(seed)
    universe = np.arange(n_samples, dtype=np.int64)
    for _attempt in range(100):
        bootstrap = rng.integers(0, n_samples, size=n_samples, dtype=np.int64)
        oob = np.setdiff1d(universe, np.unique(bootstrap), assume_unique=True)
        if oob.size:
            return OOBBootstrap(bootstrap_indices=bootstrap, oob_indices=oob)
    raise RuntimeError("could not draw a bootstrap sample with OOB rows")


def _validated_binary_targets(values: Sequence[int], *, name: str) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isin(raw, (0, 1)).all():
        raise ValueError(f"{name} must contain only original labels {{0,1}}")
    return np.asarray(raw, dtype=np.int64)


def _validated_probabilities(
    values: Sequence[float],
    *,
    name: str,
    expected_length: int | None = None,
) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if expected_length is not None and array.size != expected_length:
        raise ValueError(f"{name} must match the expected sample count")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain finite probabilities in [0,1]")
    return array


def inverse_oob_mse_weights(
    predictions: Mapping[str, OOBPrediction],
) -> OOBWeightingResult:
    """Assign normalized weights proportional to inverse OOB probability MSE.

    Models may have different OOB rows because each receives an independent
    bootstrap sample. This is why every mapping value carries both its own
    targets and probabilities instead of assuming a shared validation set.
    """

    if not predictions:
        raise ValueError("at least one model's OOB predictions are required")

    mse_by_model: dict[str, float] = {}
    inverse_errors: dict[str, float] = {}
    for model_name, prediction in predictions.items():
        if not model_name:
            raise ValueError("model names must be non-empty")
        targets = _validated_binary_targets(
            prediction.targets,
            name=f"{model_name} OOB targets",
        )
        probabilities = _validated_probabilities(
            prediction.positive_probabilities,
            name=f"{model_name} OOB probabilities",
            expected_length=targets.size,
        )
        mse = float(np.mean(np.square(targets - probabilities)))
        mse_by_model[model_name] = mse
        inverse_errors[model_name] = 1.0 / max(mse, _MSE_EPSILON)

    denominator = float(sum(inverse_errors.values()))
    weights = {
        model_name: inverse_error / denominator
        for model_name, inverse_error in inverse_errors.items()
    }
    return OOBWeightingResult(weights=weights, mse_by_model=mse_by_model)


def weighted_soft_vote(
    positive_probabilities: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
) -> FloatArray:
    """Combine model probabilities and return ensemble ``P(label=1/benign)``."""

    if not positive_probabilities:
        raise ValueError("at least one model probability vector is required")
    if set(positive_probabilities) != set(weights):
        raise ValueError("probabilities and weights must use the same model names")

    validated: dict[str, FloatArray] = {}
    expected_length: int | None = None
    for model_name, values in positive_probabilities.items():
        array = _validated_probabilities(
            values,
            name=f"{model_name} probabilities",
            expected_length=expected_length,
        )
        if expected_length is None:
            expected_length = array.size
        validated[model_name] = array

    numeric_weights: dict[str, float] = {}
    for model_name, value in weights.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError("weights must be finite and non-negative")
        numeric_weights[model_name] = numeric
    weight_sum = float(sum(numeric_weights.values()))
    if weight_sum <= 0.0:
        raise ValueError("at least one model weight must be positive")

    result = np.zeros(expected_length, dtype=float)
    for model_name, probabilities in validated.items():
        result += (numeric_weights[model_name] / weight_sum) * probabilities
    return result


def paired_correctness_ttest(
    y_true: Sequence[int],
    ensemble_predictions: Sequence[int],
    single_model_predictions: Sequence[int],
) -> PairedCorrectnessComparison:
    """Compare ensemble and best-single correctness on the same test patients."""

    targets = _validated_binary_targets(y_true, name="y_true")
    ensemble = _validated_binary_targets(
        ensemble_predictions,
        name="ensemble_predictions",
    )
    single = _validated_binary_targets(
        single_model_predictions,
        name="single_model_predictions",
    )
    if ensemble.shape != targets.shape or single.shape != targets.shape:
        raise ValueError("all paired prediction arrays must have matching shapes")

    ensemble_correct = (ensemble == targets).astype(float)
    single_correct = (single == targets).astype(float)
    difference = ensemble_correct - single_correct
    if np.all(difference == 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        test = ttest_rel(ensemble_correct, single_correct)
        statistic, p_value = float(test.statistic), float(test.pvalue)

    return PairedCorrectnessComparison(
        statistic=statistic,
        p_value=p_value,
        ensemble_accuracy=float(np.mean(ensemble_correct)),
        single_accuracy=float(np.mean(single_correct)),
        mean_correctness_difference=float(np.mean(difference)),
    )
