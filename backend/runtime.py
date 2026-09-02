"""Loaded model runtime shared by FastAPI's public endpoints.

The API process owns model fitting and inference; both frontends consume only
HTTP/JSON.  Startup happens in a background thread so ``/health`` can report a
truthful loading state while the four 100-epoch VQCs are being prepared.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import asdict
from typing import Any, Final

import numpy as np
from sklearn.datasets import load_breast_cancer

from backend.classical.baselines import (
    FEATURE_CONFIGURATIONS,
    MODEL_NAMES,
    BaselineResults,
    train_evaluate_baselines,
)
from backend.data.pipeline import (
    PreparedBreastCancerData,
    prediction_to_label,
    prepare_breast_cancer_data,
)
from backend.explain.shap_lime import ExplainabilityService
from backend.quantum.train import (
    CARRIED_SAME4_CLASSICAL_MEAN_ACCURACIES,
    MINIMUM_PHASE2_EPOCHS,
    SeedEnsembleResult,
    train_quantum_ensemble,
)


DEFAULT_MODEL_SEED: Final[int] = 42
DEFAULT_TRAINING_LIMIT: Final[int] = 20
DEFAULT_BACKGROUND_SIZE: Final[int] = 50
DEFAULT_SHAP_NSAMPLES: Final[int] = 60
DEFAULT_LIME_NUM_SAMPLES: Final[int] = 1_000

VERIFIED_PHASE2_QUANTUM: Final[dict[str, Any]] = {
    "source": "artifacts/models/phase2_benchmark_100epochs_200pool.md",
    "seeds": [42, 123, 2026],
    "vqc_epochs": 100,
    "ensemble": {
        "mean_accuracy": 0.9415,
        "accuracy_range": [0.9386, 0.9474],
    },
    "best_single": {
        "name": "qsvm_angle_4q",
        "mean_accuracy": 0.9444,
        "accuracy_range": [0.9386, 0.9474],
    },
    "paired_p_values": [0.656711, 0.656711, 0.565993],
    "finding": (
        "The ensemble validated VQC-family stability and statistically tied "
        "the strongest individual model rather than beating it."
    ),
}


class RuntimeNotReady(RuntimeError):
    """Raised when an inference endpoint is called before model loading ends."""


def _environment_integer(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _class_one_probability(estimator: Any, features: np.ndarray) -> float:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=int)
    matches = np.flatnonzero(classes == 1)
    if probabilities.shape != (1, classes.size) or matches.size != 1:
        raise RuntimeError("classical estimator must expose benign class 1")
    value = float(probabilities[0, int(matches[0])])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError("classical estimator returned an invalid probability")
    return value


def _prediction_summary(
    benign_probability: float,
    *,
    model_name: str,
    feature_count: int,
) -> dict[str, Any]:
    predicted_class = int(benign_probability >= 0.5)
    return {
        "label": prediction_to_label(predicted_class),
        "confidence": (
            benign_probability if predicted_class == 1 else 1.0 - benign_probability
        ),
        "benign_probability": benign_probability,
        "model_name": model_name,
        "feature_count": feature_count,
    }


class ModelRuntime:
    """Thread-safe holder for fitted preprocessing, quantum, and classical models."""

    def __init__(self) -> None:
        self.seed = _environment_integer(
            "QML_MODEL_SEED", DEFAULT_MODEL_SEED, minimum=0
        )
        self.vqc_epochs = _environment_integer(
            "QML_VQC_EPOCHS", MINIMUM_PHASE2_EPOCHS, minimum=MINIMUM_PHASE2_EPOCHS
        )
        self.training_limit = _environment_integer(
            "QML_TRAINING_LIMIT", DEFAULT_TRAINING_LIMIT, minimum=20
        )
        self.background_size = _environment_integer(
            "QML_EXPLANATION_BACKGROUND_SIZE",
            DEFAULT_BACKGROUND_SIZE,
            minimum=1,
        )
        self.shap_nsamples = _environment_integer(
            "QML_SHAP_NSAMPLES", DEFAULT_SHAP_NSAMPLES, minimum=1
        )
        self.lime_num_samples = _environment_integer(
            "QML_LIME_NUM_SAMPLES", DEFAULT_LIME_NUM_SAMPLES, minimum=1
        )
        self._state_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._state = "idle"
        self._error: str | None = None
        self._loader: threading.Thread | None = None

    def start_loading(self) -> None:
        """Begin fitting once; repeated lifespan starts are harmless."""

        with self._state_lock:
            if self._state in {"loading", "ready"}:
                return
            self._state = "loading"
            self._error = None
            self._loader = threading.Thread(
                target=self._load,
                name="qml-model-loader",
                daemon=True,
            )
            self._loader.start()

    def load_synchronously(self) -> None:
        """Fit in the caller thread, useful for integration checks and scripts."""

        with self._state_lock:
            if self._state == "ready":
                return
            if self._state == "loading" and self._loader is not None:
                loader = self._loader
            else:
                self._state = "loading"
                self._error = None
                loader = None
        if loader is not None:
            loader.join()
            self._require_loaded()
            return
        self._load()
        self._require_loaded()

    def _load(self) -> None:
        try:
            data = prepare_breast_cancer_data(random_state=self.seed)
            if self.training_limit > data.y_train.size:
                raise ValueError(
                    "QML_TRAINING_LIMIT cannot exceed the prepared training split"
                )
            classical = train_evaluate_baselines(data, random_state=self.seed)
            quantum = train_quantum_ensemble(
                data,
                seed=self.seed,
                vqc_epochs=self.vqc_epochs,
                training_sample_limit=self.training_limit,
                show_progress=False,
            )
            explainability = ExplainabilityService(
                quantum,
                data,
                background_size=self.background_size,
                shap_nsamples=self.shap_nsamples,
                lime_num_samples=self.lime_num_samples,
                random_state=self.seed,
            )
            dataset = load_breast_cancer()
            raw_features = np.asarray(dataset.data, dtype=float)

            # Publish a fully initialized snapshot only after every model exists.
            with self._state_lock:
                self.data = data
                self.classical = classical
                self.quantum = quantum
                self.explainability = explainability
                self.raw_features = raw_features
                self._state = "ready"
        except Exception as error:  # surfaced through /health and 503 responses
            with self._state_lock:
                self._state = "error"
                self._error = f"{type(error).__name__}: {error}"

    def _require_loaded(
        self,
    ) -> tuple[
        PreparedBreastCancerData,
        BaselineResults,
        SeedEnsembleResult,
        ExplainabilityService,
    ]:
        with self._state_lock:
            if self._state != "ready":
                detail = self._error or "models are still loading"
                raise RuntimeNotReady(detail)
            return self.data, self.classical, self.quantum, self.explainability

    def health_payload(self) -> dict[str, Any]:
        with self._state_lock:
            ready = self._state == "ready"
            return {
                "status": self._state,
                "models_loaded": ready,
                "quantum_members": len(self.quantum.members) if ready else 0,
                "classical_models": (
                    sum(len(models) for models in self.classical.values())
                    if ready
                    else 0
                ),
                "selected_features": (
                    self.data.selected_feature_names.tolist() if ready else []
                ),
                "error": self._error,
                "runtime_configuration": {
                    "seed": self.seed,
                    "vqc_epochs": self.vqc_epochs,
                    "quantum_training_limit": self.training_limit,
                    "classical_training_rows": int(self.data.y_train.size) if ready else 0,
                },
            }

    def predict_payload(self, features: list[float]) -> dict[str, Any]:
        data, classical, quantum, _explainability = self._require_loaded()
        full, selected, quantum_features = data.transform_features(
            np.asarray(features, dtype=float).reshape(1, -1)
        )
        with self._inference_lock:
            member_probabilities = [
                float(member.predict_benign_proba(quantum_features)[0])
                for member in quantum.members
            ]
            ensemble_probability = float(
                sum(
                    member.weight * probability
                    for member, probability in zip(
                        quantum.members, member_probabilities, strict=True
                    )
                )
            )
            full_estimator = classical["full_feature"][
                "logistic_regression"
            ].estimator
            same_estimator = classical["same_4_feature"][
                "logistic_regression"
            ].estimator
            full_probability = _class_one_probability(full_estimator, full)
            same_probability = _class_one_probability(same_estimator, selected)

        per_model = []
        for member, benign_probability in zip(
            quantum.members, member_probabilities, strict=True
        ):
            predicted_class = int(benign_probability >= 0.5)
            per_model.append(
                {
                    "name": member.name,
                    "paradigm": member.paradigm,
                    "label": prediction_to_label(predicted_class),
                    "confidence": (
                        benign_probability
                        if predicted_class == 1
                        else 1.0 - benign_probability
                    ),
                    "benign_probability": benign_probability,
                    "weight": member.weight,
                }
            )

        quantum_summary = _prediction_summary(
            ensemble_probability,
            model_name="six_model_oob_ensemble",
            feature_count=4,
        )
        quantum_summary.update(
            {
                "model_count": len(quantum.members),
                "per_model": per_model,
            }
        )
        full_summary = _prediction_summary(
            full_probability,
            model_name="logistic_regression",
            feature_count=30,
        )
        same_summary = _prediction_summary(
            same_probability,
            model_name="logistic_regression",
            feature_count=4,
        )
        agrees = quantum_summary["label"] == full_summary["label"]
        return {
            "quantum": quantum_summary,
            "classical": {
                "primary_configuration": "full_feature",
                "full_feature": full_summary,
                "same_4_feature": same_summary,
            },
            "agreement": {
                "agrees": agrees,
                "message": (
                    "Quantum and classical models agree on the predicted label."
                    if agrees
                    else (
                        "Quantum and classical models disagree; both results are "
                        "shown without hiding the difference."
                    )
                ),
            },
        }

    def _demo_positions(self, limit: int) -> list[int]:
        data, classical, quantum, _explainability = self._require_loaded()
        classical_predictions = classical["full_feature"][
            "logistic_regression"
        ].predictions
        disagreement = np.flatnonzero(quantum.predictions != classical_predictions)
        malignant = np.flatnonzero(data.y_test == 0)
        benign = np.flatnonzero(data.y_test == 1)
        candidates = [
            *disagreement.tolist(),
            *malignant.tolist(),
            *benign.tolist(),
            *range(data.y_test.size),
        ]
        selected: list[int] = []
        for position in candidates:
            numeric = int(position)
            if numeric not in selected:
                selected.append(numeric)
            if len(selected) == limit:
                break
        return selected

    def patients_payload(self, limit: int) -> dict[str, Any]:
        data, classical, quantum, _explainability = self._require_loaded()
        selected_indices = np.flatnonzero(data.selector.get_support())
        classical_predictions = classical["full_feature"][
            "logistic_regression"
        ].predictions
        patients = []
        for test_position in self._demo_positions(limit):
            dataset_index = int(data.test_indices[test_position])
            raw = self.raw_features[dataset_index]
            patients.append(
                {
                    "id": dataset_index,
                    "name": f"WBCD test patient {dataset_index:03d}",
                    "true_label": prediction_to_label(int(data.y_test[test_position])),
                    "features": raw.tolist(),
                    "selected_values": raw[selected_indices].tolist(),
                    "has_disagreement": bool(
                        quantum.predictions[test_position]
                        != classical_predictions[test_position]
                    ),
                }
            )
        return {
            "feature_names": data.feature_names.tolist(),
            "selected_feature_names": data.selected_feature_names.tolist(),
            "patients": patients,
        }

    def explain_payload(
        self,
        features: list[float],
        *,
        allow_slow: bool,
    ) -> dict[str, Any]:
        data, _classical, _quantum, explainability = self._require_loaded()
        _full, _selected, quantum_features = data.transform_features(
            np.asarray(features, dtype=float).reshape(1, -1)
        )
        internal_scope = "full_ensemble" if allow_slow else "fast_vqc"
        with self._inference_lock:
            explanation = explainability.explain(
                quantum_features[0],
                scope=internal_scope,
                allow_slow=allow_slow,
            )

        shap_by_name = {
            item.feature_name: item for item in explanation.shap.attributions
        }
        lime_by_name = {
            item.feature_name: item for item in explanation.lime.attributions
        }
        feature_names = list(explanation.shap.feature_names)
        top_features = []
        for shap_item in explanation.shap.top_features:
            lime_item = lime_by_name[shap_item.feature_name]
            top_features.append(
                {
                    "feature_name": shap_item.feature_name,
                    "feature_value": shap_item.feature_value,
                    "shap_value": shap_item.attribution,
                    "lime_value": lime_item.attribution,
                    "direction": shap_item.direction,
                }
            )
        return {
            "scope": "full_ensemble" if allow_slow else "vqc_fast",
            "feature_names": feature_names,
            "shap_values": [
                shap_by_name[name].attribution for name in feature_names
            ],
            "lime_values": [
                lime_by_name[name].attribution for name in feature_names
            ],
            "directions": [shap_by_name[name].direction for name in feature_names],
            "top_features": top_features,
            "expected_timing": explanation.expected_timing,
            "elapsed_seconds": explanation.elapsed_seconds,
        }

    def baselines_payload(self) -> dict[str, Any]:
        _data, classical, _quantum, _explainability = self._require_loaded()
        configurations: dict[str, Any] = {}
        for configuration in FEATURE_CONFIGURATIONS:
            model_payload = {}
            for model_name in MODEL_NAMES:
                metrics = classical[configuration][model_name].metrics
                model_payload[model_name] = asdict(metrics)
            configurations[configuration] = {
                "feature_count": 30 if configuration == "full_feature" else 4,
                "runtime_seed": self.seed,
                "models": model_payload,
            }
        return {
            "positive_class": "benign",
            "runtime_split_note": (
                "The model rows are this process's held-out seed split; judge-facing "
                "claims use the verified three-seed ranges in /metrics."
            ),
            "configurations": configurations,
        }

    def metrics_payload(self) -> dict[str, Any]:
        _data, _classical, _quantum, _explainability = self._require_loaded()
        same4 = {
            name: {
                "mean_accuracy": accuracy,
                "source": "decisions.md D-20",
            }
            for name, accuracy in CARRIED_SAME4_CLASSICAL_MEAN_ACCURACIES.items()
        }
        return {
            "quantum": VERIFIED_PHASE2_QUANTUM,
            "classical": {
                "same_4_feature_three_seed_means": same4,
                "source": "decisions.md D-20 and architecture.md section 3.7",
            },
            "positioning": (
                "Rigorously benchmarked hybrid system; no claim of quantum "
                "superiority. The ensemble tied the strongest single quantum "
                "model while reducing VQC-family instability."
            ),
        }


__all__ = ["ModelRuntime", "RuntimeNotReady", "VERIFIED_PHASE2_QUANTUM"]
