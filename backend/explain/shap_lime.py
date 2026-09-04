"""SHAP and LIME explanations for the fitted quantum ensemble.

The ordinary path deliberately explains only the VQC family. A QSVM kernel
row must be recomputed for every SHAP perturbation, so any QSVM-inclusive
scope is an explicit slow opt-in (decision D-17). All public attributions
describe the probability of sklearn class ``1`` (benign); negative values
therefore point toward class ``0`` (malignant).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable, Final, Literal, Protocol, Sequence

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ExplanationScope = Literal[
    "fast_vqc",
    "full_ensemble",
    "qsvm_diagnostic",
    "classical_full_feature",
    "classical_same_4_feature",
]
ProgressState = Literal["computing", "complete", "failed"]
AttributionDirection = Literal["toward_benign", "toward_malignant", "neutral"]

DEFAULT_BACKGROUND_SIZE: Final[int] = 50
DEFAULT_SHAP_NSAMPLES: Final[int] = 60
DEFAULT_LIME_NUM_SAMPLES: Final[int] = 1_000
DEFAULT_TOP_K: Final[int] = 3

_GENERIC_FEATURE_NAME = re.compile(
    r"^(?:pc|pca|component|feature)[ _-]?\d+$",
    flags=re.IGNORECASE,
)


class _MemberLike(Protocol):
    name: str
    paradigm: str
    weight: float

    def predict_benign_proba(self, quantum_features: FloatArray) -> FloatArray: ...


class _EnsembleLike(Protocol):
    members: Sequence[_MemberLike]


class _PreparedDataLike(Protocol):
    X_train_quantum: FloatArray
    selected_feature_names: Sequence[str]


@dataclass(frozen=True, slots=True)
class ExplanationScopeMetadata:
    """Cost and membership information suitable for API/UI presentation."""

    scope: ExplanationScope
    member_names: tuple[str, ...]
    is_slow: bool
    expected_timing: str
    expected_seconds: float


@dataclass(frozen=True, slots=True)
class ExplanationProgressEvent:
    """One lifecycle event emitted while a combined explanation is computed."""

    state: ProgressState
    scope: ExplanationScope
    message: str
    expected_timing: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class FeatureAttribution:
    """One named contribution toward benign or malignant probability."""

    feature_name: str
    feature_value: float
    attribution: float
    absolute_attribution: float
    rank: int
    direction: AttributionDirection


@dataclass(frozen=True, slots=True)
class ShapExplanation:
    """Kernel SHAP output for benign probability, with an additivity audit."""

    scope: ExplanationScope
    feature_names: tuple[str, ...]
    output_class: Literal["benign"]
    positive_label: Literal[1]
    prediction_probability: float
    expected_value: float
    reconstructed_probability: float
    additivity_residual: float
    elapsed_seconds: float
    attributions: tuple[FeatureAttribution, ...]
    top_features: tuple[FeatureAttribution, ...]


@dataclass(frozen=True, slots=True)
class LimeExplanation:
    """LIME local-surrogate output for sklearn class 1 (benign).

    ``attributions`` are each local coefficient multiplied by the patient's
    standardized displacement from LIME's training mean.  This turns a slope
    into an observed-patient contribution, making its direction comparable to
    SHAP's background-relative contribution.
    """

    scope: ExplanationScope
    feature_names: tuple[str, ...]
    output_class: Literal["benign"]
    positive_label: Literal[1]
    prediction_probability: float
    surrogate_intercept: float | None
    surrogate_prediction: float | None
    reconstructed_surrogate_prediction: float | None
    surrogate_additivity_residual: float | None
    elapsed_seconds: float
    attributions: tuple[FeatureAttribution, ...]
    top_features: tuple[FeatureAttribution, ...]


@dataclass(frozen=True, slots=True)
class AttributionSignComparison:
    """Direction comparison for a feature ranked highly by both methods."""

    feature_name: str
    shap_direction: AttributionDirection
    lime_direction: AttributionDirection
    agrees: bool | None


@dataclass(frozen=True, slots=True)
class AttributionCrossCheck:
    """Descriptive SHAP/LIME agreement; disagreement is valid information."""

    top_k: int
    top_feature_overlap: tuple[str, ...]
    top_feature_overlap_count: int
    top_feature_overlap_ratio: float
    sign_comparisons: tuple[AttributionSignComparison, ...]
    comparable_sign_count: int
    sign_agreement_count: int
    sign_agreement_ratio: float | None


@dataclass(frozen=True, slots=True)
class CombinedExplanation:
    """SHAP result, LIME cross-check, and truthful lifecycle/timing metadata."""

    status: Literal["complete"]
    scope: ExplanationScope
    member_names: tuple[str, ...]
    expected_timing: str
    elapsed_seconds: float
    prediction_probability: float
    shap: ShapExplanation
    lime: LimeExplanation
    cross_check: AttributionCrossCheck
    progress_events: tuple[ExplanationProgressEvent, ...]


ProgressCallback = Callable[[ExplanationProgressEvent], None]


def _validated_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _validated_feature_names(
    values: Sequence[str],
    *,
    expected_count: int,
) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values)
    if len(names) != expected_count or not all(names):
        raise ValueError(
            f"selected_feature_names must contain exactly {expected_count} names"
        )
    if len(set(names)) != len(names):
        raise ValueError("selected_feature_names must be unique")
    if any(_GENERIC_FEATURE_NAME.fullmatch(name) for name in names):
        raise ValueError(
            "SHAP and LIME require real clinical feature names, not generic "
            "PCA/component/feature-number labels"
        )
    return names


def _as_feature_matrix(
    values: ArrayLike,
    *,
    n_features: int,
    name: str,
) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != n_features:
        raise ValueError(f"{name} must have shape (n_samples, {n_features})")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite rows")
    return np.asarray(matrix, dtype=float)


def _as_single_patient(
    values: ArrayLike,
    *,
    n_features: int,
) -> FloatArray:
    patient = _as_feature_matrix(values, n_features=n_features, name="patient")
    if patient.shape[0] != 1:
        raise ValueError("one explanation requires exactly one patient row")
    return patient


def _direction(value: float, *, tolerance: float = 1e-12) -> AttributionDirection:
    if value > tolerance:
        return "toward_benign"
    if value < -tolerance:
        return "toward_malignant"
    return "neutral"


def _structured_attributions(
    feature_names: tuple[str, ...],
    feature_values: FloatArray,
    contributions: FloatArray,
    *,
    top_k: int,
) -> tuple[tuple[FeatureAttribution, ...], tuple[FeatureAttribution, ...]]:
    order = np.argsort(-np.abs(contributions), kind="stable")
    ranks = np.empty(len(feature_names), dtype=int)
    ranks[order] = np.arange(1, len(feature_names) + 1)
    attributions = tuple(
        FeatureAttribution(
            feature_name=feature_name,
            feature_value=float(feature_values[index]),
            attribution=float(contributions[index]),
            absolute_attribution=float(abs(contributions[index])),
            rank=int(ranks[index]),
            direction=_direction(float(contributions[index])),
        )
        for index, feature_name in enumerate(feature_names)
    )
    by_rank = sorted(attributions, key=lambda item: item.rank)
    return attributions, tuple(by_rank[:top_k])


def _single_shap_vector(raw_values: object, *, n_features: int) -> FloatArray:
    values = raw_values
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("SHAP returned multiple outputs for a scalar predictor")
        values = values[0]
    array = np.asarray(values, dtype=float)
    while array.ndim > 1 and 1 in array.shape:
        singleton_axis = next(
            index for index, size in enumerate(array.shape) if size == 1
        )
        array = np.squeeze(array, axis=singleton_axis)
    if array.ndim != 1 or array.size != n_features:
        raise ValueError(
            "SHAP must return one attribution per selected clinical feature"
        )
    if not np.isfinite(array).all():
        raise ValueError("SHAP returned non-finite attributions")
    return np.asarray(array, dtype=float)


def _single_numeric(value: object, *, name: str) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"{name} must be one finite scalar")
    return float(array[0])


def _optional_mapping_value(value: object, *, class_index: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if class_index not in value:
            return None
        value = value[class_index]
    array = np.asarray(value, dtype=float).reshape(-1)
    if not array.size or not np.isfinite(array[0]):
        return None
    return float(array[0])


def cross_check_attributions(
    shap_explanation: ShapExplanation,
    lime_explanation: LimeExplanation,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> AttributionCrossCheck:
    """Compare top-feature overlap and direction without requiring agreement."""

    top_count = min(
        _validated_positive_integer(top_k, name="top_k"),
        len(shap_explanation.feature_names),
    )
    shap_top = shap_explanation.top_features[:top_count]
    lime_top = lime_explanation.top_features[:top_count]
    lime_names = {item.feature_name for item in lime_top}
    overlap = tuple(
        item.feature_name for item in shap_top if item.feature_name in lime_names
    )
    shap_by_name = {item.feature_name: item for item in shap_top}
    lime_by_name = {item.feature_name: item for item in lime_top}
    sign_comparisons: list[AttributionSignComparison] = []
    comparable_count = 0
    agreement_count = 0
    for feature_name in overlap:
        shap_direction = shap_by_name[feature_name].direction
        lime_direction = lime_by_name[feature_name].direction
        if "neutral" in (shap_direction, lime_direction):
            agrees: bool | None = None
        else:
            comparable_count += 1
            agrees = shap_direction == lime_direction
            agreement_count += int(agrees)
        sign_comparisons.append(
            AttributionSignComparison(
                feature_name=feature_name,
                shap_direction=shap_direction,
                lime_direction=lime_direction,
                agrees=agrees,
            )
        )
    agreement_ratio = agreement_count / comparable_count if comparable_count else None
    return AttributionCrossCheck(
        top_k=top_count,
        top_feature_overlap=overlap,
        top_feature_overlap_count=len(overlap),
        top_feature_overlap_ratio=len(overlap) / top_count,
        sign_comparisons=tuple(sign_comparisons),
        comparable_sign_count=comparable_count,
        sign_agreement_count=agreement_count,
        sign_agreement_ratio=agreement_ratio,
    )


class ExplainabilityService:
    """Explain a fitted Phase 2 ensemble in deterministic, UI-ready scopes.

    ``prepared`` is the fitted split that produced the ensemble. Its
    ``selected_feature_names`` are copied verbatim into both explainers, and
    its quantum-range training matrix supplies the deterministic background.
    """

    def __init__(
        self,
        ensemble_result: _EnsembleLike,
        prepared: _PreparedDataLike,
        *,
        background_size: int = DEFAULT_BACKGROUND_SIZE,
        shap_nsamples: int = DEFAULT_SHAP_NSAMPLES,
        lime_num_samples: int = DEFAULT_LIME_NUM_SAMPLES,
        random_state: int = 42,
    ) -> None:
        members = tuple(ensemble_result.members)
        if not members:
            raise ValueError("ensemble_result must contain at least one member")
        if len({str(member.name) for member in members}) != len(members):
            raise ValueError("ensemble member names must be unique")
        for member in members:
            if member.paradigm not in ("VQC", "QSVM"):
                raise ValueError("ensemble members must be VQC or QSVM models")
            weight = float(member.weight)
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    "ensemble member weights must be finite and non-negative"
                )

        raw_background = np.asarray(prepared.X_train_quantum, dtype=float)
        if raw_background.ndim != 2 or raw_background.shape[1] == 0:
            raise ValueError("prepared.X_train_quantum must be a feature matrix")
        self.feature_names = _validated_feature_names(
            prepared.selected_feature_names,
            expected_count=raw_background.shape[1],
        )
        background = _as_feature_matrix(
            raw_background,
            n_features=len(self.feature_names),
            name="prepared.X_train_quantum",
        )
        requested_background_size = _validated_positive_integer(
            background_size,
            name="background_size",
        )
        selected_background_size = min(requested_background_size, background.shape[0])
        if selected_background_size < background.shape[0]:
            rng = np.random.default_rng(int(random_state))
            indices = np.sort(
                rng.choice(
                    background.shape[0],
                    size=selected_background_size,
                    replace=False,
                )
            )
            background = background[indices]

        self._members = members
        self.background = np.asarray(background, dtype=float).copy()
        self.shap_nsamples = _validated_positive_integer(
            shap_nsamples,
            name="shap_nsamples",
        )
        self.lime_num_samples = _validated_positive_integer(
            lime_num_samples,
            name="lime_num_samples",
        )
        self.random_state = int(random_state)

    def scope_metadata(
        self,
        scope: ExplanationScope = "fast_vqc",
        *,
        allow_slow: bool = False,
    ) -> ExplanationScopeMetadata:
        """Return the selected members and enforce explicit slow-scope consent."""

        if scope == "fast_vqc":
            selected = tuple(
                member for member in self._members if member.paradigm == "VQC"
            )
            is_slow = False
            expected_timing = "about 7 seconds"
            expected_seconds = 7.0
        elif scope == "full_ensemble":
            selected = self._members
            is_slow = True
            expected_timing = "at least 70 seconds"
            expected_seconds = 70.0
        elif scope == "qsvm_diagnostic":
            selected = tuple(
                member for member in self._members if member.paradigm == "QSVM"
            )
            is_slow = True
            expected_timing = "at least 70 seconds"
            expected_seconds = 70.0
        else:
            raise ValueError(f"unknown explanation scope: {scope!r}")

        if is_slow and not allow_slow:
            raise PermissionError(
                f"{scope} is QSVM-inclusive and may take at least 70 seconds; "
                "pass allow_slow=True to opt in explicitly"
            )
        if not selected:
            raise ValueError(f"no ensemble members are available for scope {scope!r}")
        if sum(float(member.weight) for member in selected) <= 0.0:
            raise ValueError(f"scope {scope!r} has no positive member weight")
        return ExplanationScopeMetadata(
            scope=scope,
            member_names=tuple(str(member.name) for member in selected),
            is_slow=is_slow,
            expected_timing=expected_timing,
            expected_seconds=expected_seconds,
        )

    def _scope_predictor(
        self,
        scope: ExplanationScope,
        *,
        allow_slow: bool,
    ) -> tuple[Callable[[ArrayLike], FloatArray], ExplanationScopeMetadata]:
        metadata = self.scope_metadata(scope, allow_slow=allow_slow)
        selected_names = set(metadata.member_names)
        members = tuple(
            member for member in self._members if member.name in selected_names
        )
        raw_weights = np.asarray([float(member.weight) for member in members])
        weights = raw_weights / raw_weights.sum()

        def predict(features: ArrayLike) -> FloatArray:
            X = _as_feature_matrix(
                features,
                n_features=len(self.feature_names),
                name="quantum features",
            )
            result = np.zeros(X.shape[0], dtype=float)
            for member, weight in zip(members, weights, strict=True):
                probabilities = np.asarray(
                    member.predict_benign_proba(X),
                    dtype=float,
                )
                if probabilities.shape != (X.shape[0],):
                    raise ValueError(
                        f"{member.name} must return one benign probability per row"
                    )
                if not np.isfinite(probabilities).all() or np.any(
                    (probabilities < 0.0) | (probabilities > 1.0)
                ):
                    raise ValueError(
                        f"{member.name} returned invalid benign probabilities"
                    )
                result += float(weight) * probabilities
            return result

        return predict, metadata

    def predict_benign_probability(
        self,
        features: ArrayLike,
        *,
        scope: ExplanationScope = "fast_vqc",
        allow_slow: bool = False,
    ) -> FloatArray:
        """Predict benign probability in one scope; defaults to VQCs only."""

        predictor, _metadata = self._scope_predictor(
            scope,
            allow_slow=allow_slow,
        )
        return predictor(features)

    def explain_shap(
        self,
        patient_quantum: ArrayLike,
        *,
        scope: ExplanationScope = "fast_vqc",
        allow_slow: bool = False,
        nsamples: int | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> ShapExplanation:
        """Run Kernel SHAP against benign probability for exactly one patient."""

        patient = _as_single_patient(
            patient_quantum,
            n_features=len(self.feature_names),
        )
        sample_count = (
            self.shap_nsamples
            if nsamples is None
            else _validated_positive_integer(nsamples, name="nsamples")
        )
        top_count = min(
            _validated_positive_integer(top_k, name="top_k"),
            len(self.feature_names),
        )
        predictor, metadata = self._scope_predictor(
            scope,
            allow_slow=allow_slow,
        )
        started = time.perf_counter()
        explainer = shap.KernelExplainer(
            predictor,
            self.background,
            feature_names=list(self.feature_names),
            link="identity",
        )
        raw_values = explainer.shap_values(
            patient,
            nsamples=sample_count,
            silent=True,
        )
        values = _single_shap_vector(raw_values, n_features=len(self.feature_names))
        expected_value = _single_numeric(
            explainer.expected_value,
            name="SHAP expected_value",
        )
        prediction_probability = float(predictor(patient)[0])
        reconstructed = float(expected_value + values.sum())
        attributions, top_features = _structured_attributions(
            self.feature_names,
            patient[0],
            values,
            top_k=top_count,
        )
        return ShapExplanation(
            scope=metadata.scope,
            feature_names=self.feature_names,
            output_class="benign",
            positive_label=1,
            prediction_probability=prediction_probability,
            expected_value=expected_value,
            reconstructed_probability=reconstructed,
            additivity_residual=float(prediction_probability - reconstructed),
            elapsed_seconds=time.perf_counter() - started,
            attributions=attributions,
            top_features=top_features,
        )

    def explain_lime(
        self,
        patient_quantum: ArrayLike,
        *,
        scope: ExplanationScope = "fast_vqc",
        allow_slow: bool = False,
        num_samples: int | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> LimeExplanation:
        """Run LIME as an independent local cross-check for benign probability."""

        patient = _as_single_patient(
            patient_quantum,
            n_features=len(self.feature_names),
        )
        sample_count = (
            self.lime_num_samples
            if num_samples is None
            else _validated_positive_integer(num_samples, name="num_samples")
        )
        top_count = min(
            _validated_positive_integer(top_k, name="top_k"),
            len(self.feature_names),
        )
        predictor, metadata = self._scope_predictor(
            scope,
            allow_slow=allow_slow,
        )

        def predict_two_class(features: ArrayLike) -> FloatArray:
            benign = predictor(features)
            return np.column_stack((1.0 - benign, benign))

        started = time.perf_counter()
        explainer = LimeTabularExplainer(
            self.background,
            mode="classification",
            feature_names=list(self.feature_names),
            class_names=["malignant", "benign"],
            discretize_continuous=False,
            random_state=self.random_state,
        )
        explanation = explainer.explain_instance(
            patient[0],
            predict_two_class,
            labels=(1,),
            num_features=len(self.feature_names),
            num_samples=sample_count,
        )
        mapping = explanation.as_map()
        if 1 not in mapping:
            raise ValueError("LIME did not return an explanation for benign class 1")
        scaler = getattr(explainer, "scaler", None)
        scaler_mean = np.asarray(getattr(scaler, "mean_", None), dtype=float)
        scaler_scale = np.asarray(getattr(scaler, "scale_", None), dtype=float)
        expected_shape = (len(self.feature_names),)
        if (
            scaler_mean.shape != expected_shape
            or scaler_scale.shape != expected_shape
            or not np.isfinite(scaler_mean).all()
            or not np.isfinite(scaler_scale).all()
            or np.any(scaler_scale <= 0.0)
        ):
            raise ValueError("LIME returned invalid feature scaling metadata")
        patient_scaled = (patient[0] - scaler_mean) / scaler_scale
        values = np.zeros(len(self.feature_names), dtype=float)
        for feature_index, contribution in mapping[1]:
            index = int(feature_index)
            if not 0 <= index < len(self.feature_names):
                raise ValueError("LIME returned an out-of-range feature index")
            numeric = float(contribution)
            if not np.isfinite(numeric):
                raise ValueError("LIME returned a non-finite attribution")
            values[index] = numeric * patient_scaled[index]
        prediction_probability = float(predictor(patient)[0])
        surrogate_intercept = _optional_mapping_value(
            getattr(explanation, "intercept", None),
            class_index=1,
        )
        surrogate_prediction = _optional_mapping_value(
            getattr(explanation, "local_pred", None),
            class_index=1,
        )
        reconstructed_surrogate = (
            None
            if surrogate_intercept is None
            else float(surrogate_intercept + values.sum())
        )
        surrogate_residual = (
            None
            if surrogate_prediction is None or reconstructed_surrogate is None
            else float(surrogate_prediction - reconstructed_surrogate)
        )
        attributions, top_features = _structured_attributions(
            self.feature_names,
            patient[0],
            values,
            top_k=top_count,
        )
        return LimeExplanation(
            scope=metadata.scope,
            feature_names=self.feature_names,
            output_class="benign",
            positive_label=1,
            prediction_probability=prediction_probability,
            surrogate_intercept=surrogate_intercept,
            surrogate_prediction=surrogate_prediction,
            reconstructed_surrogate_prediction=reconstructed_surrogate,
            surrogate_additivity_residual=surrogate_residual,
            elapsed_seconds=time.perf_counter() - started,
            attributions=attributions,
            top_features=top_features,
        )

    def explain(
        self,
        patient_quantum: ArrayLike,
        *,
        scope: ExplanationScope = "fast_vqc",
        allow_slow: bool = False,
        nsamples: int | None = None,
        lime_num_samples: int | None = None,
        top_k: int = DEFAULT_TOP_K,
        progress: ProgressCallback | None = None,
    ) -> CombinedExplanation:
        """Run SHAP plus LIME, emitting computing/complete/failed events."""

        metadata = self.scope_metadata(scope, allow_slow=allow_slow)
        events: list[ExplanationProgressEvent] = []
        started = time.perf_counter()

        def emit(state: ProgressState, message: str) -> None:
            event = ExplanationProgressEvent(
                state=state,
                scope=metadata.scope,
                message=message,
                expected_timing=metadata.expected_timing,
                elapsed_seconds=time.perf_counter() - started,
            )
            events.append(event)
            if progress is not None:
                progress(event)

        emit(
            "computing",
            "computing explanation... "
            f"{metadata.scope} is expected to take {metadata.expected_timing}",
        )
        try:
            shap_result = self.explain_shap(
                patient_quantum,
                scope=scope,
                allow_slow=allow_slow,
                nsamples=nsamples,
                top_k=top_k,
            )
            lime_result = self.explain_lime(
                patient_quantum,
                scope=scope,
                allow_slow=allow_slow,
                num_samples=lime_num_samples,
                top_k=top_k,
            )
            cross_check = cross_check_attributions(
                shap_result,
                lime_result,
                top_k=top_k,
            )
        except Exception as error:
            emit("failed", f"explanation failed: {error}")
            raise

        elapsed = time.perf_counter() - started
        emit("complete", f"explanation complete in {elapsed:.2f} seconds")
        return CombinedExplanation(
            status="complete",
            scope=metadata.scope,
            member_names=metadata.member_names,
            expected_timing=metadata.expected_timing,
            elapsed_seconds=elapsed,
            prediction_probability=shap_result.prediction_probability,
            shap=shap_result,
            lime=lime_result,
            cross_check=cross_check,
            progress_events=tuple(events),
        )


class ClassicalExplainabilityService:
    """SHAP + LIME for one fitted classical probability estimator.

    The implementation reuses the validated model-agnostic explainability path
    while retaining the classical scope and all original clinical feature
    names in the public result.
    """

    def __init__(
        self,
        estimator: object,
        training_features: ArrayLike,
        feature_names: Sequence[str],
        *,
        scope: Literal["classical_full_feature", "classical_same_4_feature"],
        background_size: int = DEFAULT_BACKGROUND_SIZE,
        shap_nsamples: int = DEFAULT_SHAP_NSAMPLES,
        lime_num_samples: int = DEFAULT_LIME_NUM_SAMPLES,
        random_state: int = 42,
    ) -> None:
        if scope not in ("classical_full_feature", "classical_same_4_feature"):
            raise ValueError("invalid classical explanation scope")
        classes = np.asarray(getattr(estimator, "classes_", ()), dtype=int)
        matches = np.flatnonzero(classes == 1)
        if matches.size != 1 or not hasattr(estimator, "predict_proba"):
            raise ValueError("classical explainer requires benign-class probabilities")
        class_index = int(matches[0])

        class Member:
            name = scope
            paradigm = "VQC"
            weight = 1.0

            @staticmethod
            def predict_benign_proba(features: FloatArray) -> FloatArray:
                probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
                return probabilities[:, class_index]

        matrix = np.asarray(training_features, dtype=float)
        names = _validated_feature_names(feature_names, expected_count=matrix.shape[1])
        self.scope = scope
        self._delegate = ExplainabilityService(
            SimpleNamespace(members=(Member(),)),
            SimpleNamespace(X_train_quantum=matrix, selected_feature_names=names),
            background_size=background_size,
            shap_nsamples=shap_nsamples,
            lime_num_samples=lime_num_samples,
            random_state=random_state,
        )

    def explain(self, patient_features: ArrayLike) -> CombinedExplanation:
        """Return attribution only metadata for the configured classical view."""

        result = self._delegate.explain(patient_features, scope="fast_vqc")
        timing = "a few seconds"
        shap_result = replace(result.shap, scope=self.scope)
        lime_result = replace(result.lime, scope=self.scope)
        events = tuple(
            replace(event, scope=self.scope, expected_timing=timing)
            for event in result.progress_events
        )
        return replace(
            result,
            scope=self.scope,
            member_names=(self.scope,),
            expected_timing=timing,
            shap=shap_result,
            lime=lime_result,
            progress_events=events,
        )


__all__ = [
    "AttributionCrossCheck",
    "AttributionSignComparison",
    "CombinedExplanation",
    "ClassicalExplainabilityService",
    "DEFAULT_BACKGROUND_SIZE",
    "DEFAULT_LIME_NUM_SAMPLES",
    "DEFAULT_SHAP_NSAMPLES",
    "DEFAULT_TOP_K",
    "ExplainabilityService",
    "ExplanationProgressEvent",
    "ExplanationScope",
    "ExplanationScopeMetadata",
    "FeatureAttribution",
    "LimeExplanation",
    "ShapExplanation",
    "cross_check_attributions",
]
