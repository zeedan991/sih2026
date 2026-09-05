"""Loaded model runtime shared by FastAPI's public endpoints.

The API process owns model fitting and inference; both frontends consume only
HTTP/JSON.  Startup happens in a background thread so ``/health`` can report a
truthful loading state while the four 100-epoch VQCs are being prepared.
"""

from __future__ import annotations

import math
import os
import threading
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import numpy as np

from backend.classical.baselines import (
    FEATURE_CONFIGURATIONS,
    MODEL_NAMES,
    BaselineResults,
    evaluate_classification,
    train_evaluate_baselines,
)
from backend.data.diseases import (
    DEFAULT_DISEASE_ID,
    DISEASES,
    DiseaseDefinition,
    disease_definition,
)
from backend.explain.shap_lime import ClassicalExplainabilityService, ExplainabilityService
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
DEFAULT_RUNTIME_MANIFEST: Final[Path] = (
    Path(__file__).resolve().parent.parent / "artifacts" / "models" / "runtime_manifest.json"
)

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
    class_one_probability: float,
    *,
    definition: DiseaseDefinition = DISEASES[DEFAULT_DISEASE_ID],
    model_name: str,
    feature_count: int,
) -> dict[str, Any]:
    predicted_class = int(class_one_probability >= 0.5)
    result = {
        "label": definition.label_for_class(predicted_class),
        "confidence": (
            class_one_probability
            if predicted_class == 1
            else 1.0 - class_one_probability
        ),
        "class_one_probability": class_one_probability,
        "class_probabilities": {
            definition.class_labels[0]: 1.0 - class_one_probability,
            definition.class_labels[1]: class_one_probability,
        },
        "model_name": model_name,
        "feature_count": feature_count,
    }
    # Preserve the Phase 4 breast-cancer contract for existing clients while
    # avoiding the clinically incorrect word "benign" in other modules.
    if definition.disease_id == "breast_cancer":
        result["benign_probability"] = class_one_probability
    return result


@dataclass(slots=True)
class DiseaseRuntimeBundle:
    definition: DiseaseDefinition
    data: Any
    classical: BaselineResults
    quantum: SeedEnsembleResult
    explainability: ExplainabilityService
    classical_explainability: dict[str, ClassicalExplainabilityService]
    raw_features: np.ndarray


class ModelRuntime:
    """Thread-safe holder for fitted preprocessing, quantum, and classical models."""

    def __init__(self) -> None:
        manifest_path = Path(os.getenv("QML_RUNTIME_MANIFEST", str(DEFAULT_RUNTIME_MANIFEST)))
        if not manifest_path.exists():
            raise ValueError(f"runtime manifest does not exist: {manifest_path}")
        self.runtime_manifest_path = manifest_path
        self.runtime_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_seed = int(self.runtime_manifest["split_seed"])
        manifest_epochs = int(self.runtime_manifest["vqc_epochs"])
        manifest_limit = int(self.runtime_manifest["quantum_training_limit"])
        self.quantum_device = os.getenv(
            "QML_QUANTUM_DEVICE", str(self.runtime_manifest["quantum_device"])
        ).strip()
        if not self.quantum_device:
            raise ValueError("QML_QUANTUM_DEVICE must be non-empty")
        self.seed = _environment_integer(
            "QML_MODEL_SEED", manifest_seed, minimum=0
        )
        self.vqc_epochs = _environment_integer(
            "QML_VQC_EPOCHS", manifest_epochs, minimum=MINIMUM_PHASE2_EPOCHS
        )
        self.training_limit = _environment_integer(
            "QML_TRAINING_LIMIT", manifest_limit, minimum=20
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
            bundles: dict[str, DiseaseRuntimeBundle] = {}
            for definition in DISEASES.values():
                data = definition.prepare_data(random_state=self.seed)
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
                    device_name=self.quantum_device,
                )
                explainability = ExplainabilityService(
                    quantum,
                    data,
                    background_size=self.background_size,
                    shap_nsamples=self.shap_nsamples,
                    lime_num_samples=self.lime_num_samples,
                    random_state=self.seed,
                )
                classical_explainability = {
                    "classical_full_feature": ClassicalExplainabilityService(
                        classical["full_feature"]["logistic_regression"].estimator,
                        data.X_train_full,
                        data.feature_names,
                        scope="classical_full_feature",
                        background_size=self.background_size,
                        shap_nsamples=self.shap_nsamples,
                        lime_num_samples=self.lime_num_samples,
                        random_state=self.seed,
                    ),
                    "classical_same_4_feature": ClassicalExplainabilityService(
                        classical["same_4_feature"]["logistic_regression"].estimator,
                        data.X_train_selected,
                        data.selected_feature_names,
                        scope="classical_same_4_feature",
                        background_size=self.background_size,
                        shap_nsamples=self.shap_nsamples,
                        lime_num_samples=self.lime_num_samples,
                        random_state=self.seed,
                    ),
                }
                bundles[definition.disease_id] = DiseaseRuntimeBundle(
                    definition=definition,
                    data=data,
                    classical=classical,
                    quantum=quantum,
                    explainability=explainability,
                    classical_explainability=classical_explainability,
                    raw_features=definition.load_raw_features(),
                )

            # Publish a fully initialized snapshot only after every model exists.
            with self._state_lock:
                self.bundles = bundles
                # Backwards-compatible aliases keep the verified WBCD
                # evaluation utilities stable while the API becomes modular.
                default = bundles[DEFAULT_DISEASE_ID]
                self.data = default.data
                self.classical = default.classical
                self.quantum = default.quantum
                self.explainability = default.explainability
                self.classical_explainability = default.classical_explainability
                self.raw_features = default.raw_features
                self._state = "ready"
        except Exception as error:  # surfaced through /health and 503 responses
            with self._state_lock:
                self._state = "error"
                self._error = f"{type(error).__name__}: {error}"

    def _require_loaded(
        self, disease_id: str = DEFAULT_DISEASE_ID
    ) -> DiseaseRuntimeBundle:
        with self._state_lock:
            if self._state != "ready":
                detail = self._error or "models are still loading"
                raise RuntimeNotReady(detail)
            try:
                return self.bundles[disease_id]
            except KeyError as error:
                raise ValueError(
                    f"unknown disease_id {disease_id!r}; choose one of {', '.join(DISEASES)}"
                ) from error

    def health_payload(self) -> dict[str, Any]:
        with self._state_lock:
            ready = self._state == "ready"
            project_root = Path(__file__).resolve().parent.parent
            try:
                manifest_display = str(
                    self.runtime_manifest_path.relative_to(project_root)
                )
            except ValueError:
                manifest_display = str(self.runtime_manifest_path)
            return {
                "status": self._state,
                "models_loaded": ready,
                "quantum_members": (
                    sum(len(bundle.quantum.members) for bundle in self.bundles.values())
                    if ready
                    else 0
                ),
                "classical_models": (
                    sum(
                        len(models)
                        for bundle in self.bundles.values()
                        for models in bundle.classical.values()
                    )
                    if ready
                    else 0
                ),
                "selected_features": (
                    self.data.selected_feature_names.tolist() if ready else []
                ),
                "error": self._error,
                "disease_modules": (
                    [
                        {
                            "disease_id": bundle.definition.disease_id,
                            "title": bundle.definition.title,
                            "dataset_name": bundle.definition.dataset_name,
                            "feature_count": int(bundle.data.feature_names.size),
                            "classical_training_rows": int(bundle.data.y_train.size),
                            "selected_features": bundle.data.selected_feature_names.tolist(),
                            "quantum_members": len(bundle.quantum.members),
                        }
                        for bundle in self.bundles.values()
                    ]
                    if ready
                    else []
                ),
                "runtime_configuration": {
                    "configuration_id": self.runtime_manifest["configuration_id"],
                    "manifest": manifest_display,
                    "loading_mode": self.runtime_manifest["loading_mode"],
                    "quantum_device": self.quantum_device,
                    "seed": self.seed,
                    "vqc_epochs": self.vqc_epochs,
                    "quantum_training_limit": self.training_limit,
                    "classical_training_rows": int(self.data.y_train.size) if ready else 0,
                },
            }

    def diseases_payload(self) -> dict[str, Any]:
        """Return the installed module registry and dynamic input schemas."""

        modules = []
        for disease_id, definition in DISEASES.items():
            bundle = self._require_loaded(disease_id)
            minimum = bundle.raw_features.min(axis=0)
            maximum = bundle.raw_features.max(axis=0)
            modules.append(
                {
                    "disease_id": disease_id,
                    "title": definition.title,
                    "short_title": definition.short_title,
                    "domain": definition.domain,
                    "dataset_name": definition.dataset_name,
                    "dataset_source": definition.dataset_source,
                    "dataset_license": definition.dataset_license,
                    "description": definition.description,
                    "positive_class_name": definition.positive_class_name,
                    "class_labels": list(definition.class_labels),
                    "feature_names": bundle.data.feature_names.tolist(),
                    "selected_feature_names": bundle.data.selected_feature_names.tolist(),
                    "feature_schema": [
                        {
                            "name": str(name),
                            "kind": definition.feature_kinds[index],
                            "observed_min": float(minimum[index]),
                            "observed_max": float(maximum[index]),
                        }
                        for index, name in enumerate(bundle.data.feature_names)
                    ],
                }
            )
        return {"default_disease_id": DEFAULT_DISEASE_ID, "modules": modules}

    @staticmethod
    def _validated_raw_row(
        bundle: DiseaseRuntimeBundle, features: list[float]
    ) -> np.ndarray:
        raw = np.asarray(features, dtype=float)
        if raw.ndim != 1 or raw.size != bundle.data.feature_names.size:
            raise ValueError(
                f"{bundle.definition.disease_id} requires exactly "
                f"{bundle.data.feature_names.size} feature values"
            )
        if not np.isfinite(raw).all():
            raise ValueError("features must contain only finite values")
        invalid_binary = [
            str(bundle.data.feature_names[index])
            for index, kind in enumerate(bundle.definition.feature_kinds)
            if kind == "binary" and raw[index] not in (0.0, 1.0)
        ]
        if invalid_binary:
            raise ValueError(
                "binary questionnaire features must be encoded as 0 or 1: "
                + ", ".join(invalid_binary)
            )
        return raw

    def predict_payload(
        self, features: list[float], *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]:
        bundle = self._require_loaded(disease_id)
        definition = bundle.definition
        data, classical, quantum = bundle.data, bundle.classical, bundle.quantum
        raw = self._validated_raw_row(bundle, features)
        full, selected, quantum_features = data.transform_features(
            raw.reshape(1, -1)
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
        for member, class_one_probability in zip(
            quantum.members, member_probabilities, strict=True
        ):
            predicted_class = int(class_one_probability >= 0.5)
            summary = _prediction_summary(
                class_one_probability,
                definition=definition,
                model_name=member.name,
                feature_count=4,
            )
            per_model.append(
                {
                    "name": member.name,
                    "paradigm": member.paradigm,
                    "label": definition.label_for_class(predicted_class),
                    "confidence": summary["confidence"],
                    "class_one_probability": class_one_probability,
                    "class_probabilities": summary["class_probabilities"],
                    "weight": member.weight,
                    **(
                        {"benign_probability": class_one_probability}
                        if disease_id == "breast_cancer"
                        else {}
                    ),
                }
            )

        quantum_summary = _prediction_summary(
            ensemble_probability,
            definition=definition,
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
            definition=definition,
            model_name="logistic_regression",
            feature_count=int(data.feature_names.size),
        )
        same_summary = _prediction_summary(
            same_probability,
            definition=definition,
            model_name="logistic_regression",
            feature_count=4,
        )
        agrees = quantum_summary["label"] == full_summary["label"]
        return {
            "disease_id": disease_id,
            "disease_title": definition.title,
            "dataset_name": definition.dataset_name,
            "class_labels": list(definition.class_labels),
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

    def _demo_positions(self, limit: int, disease_id: str) -> list[int]:
        bundle = self._require_loaded(disease_id)
        data, classical, quantum = bundle.data, bundle.classical, bundle.quantum
        classical_predictions = classical["full_feature"][
            "logistic_regression"
        ].predictions
        disagreement = np.flatnonzero(quantum.predictions != classical_predictions)
        condition_present = np.flatnonzero(data.y_test == 0)
        condition_absent = np.flatnonzero(data.y_test == 1)
        candidates = [
            *disagreement.tolist(),
            *condition_present.tolist(),
            *condition_absent.tolist(),
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

    def patients_payload(
        self, limit: int, *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]:
        bundle = self._require_loaded(disease_id)
        definition = bundle.definition
        data, classical, quantum = bundle.data, bundle.classical, bundle.quantum
        selected_indices = np.flatnonzero(data.selector.get_support())
        classical_predictions = classical["full_feature"][
            "logistic_regression"
        ].predictions
        patients = []
        for test_position in self._demo_positions(limit, disease_id):
            dataset_index = int(data.test_indices[test_position])
            raw = bundle.raw_features[dataset_index]
            patients.append(
                {
                    "id": dataset_index,
                    "name": f"{definition.record_prefix} test record {dataset_index:03d}",
                    "true_label": definition.label_for_class(
                        int(data.y_test[test_position])
                    ),
                    "features": raw.tolist(),
                    "selected_values": raw[selected_indices].tolist(),
                    "has_disagreement": bool(
                        quantum.predictions[test_position]
                        != classical_predictions[test_position]
                    ),
                }
            )
        return {
            "disease_id": disease_id,
            "disease_title": definition.title,
            "dataset_name": definition.dataset_name,
            "class_labels": list(definition.class_labels),
            "feature_names": data.feature_names.tolist(),
            "selected_feature_names": data.selected_feature_names.tolist(),
            "patients": patients,
        }

    def explain_payload(
        self,
        features: list[float],
        *,
        model: str,
        allow_slow: bool,
        disease_id: str = DEFAULT_DISEASE_ID,
    ) -> dict[str, Any]:
        bundle = self._require_loaded(disease_id)
        data = bundle.data
        quantum_explainability = bundle.explainability
        raw = self._validated_raw_row(bundle, features)
        full, selected, quantum_features = data.transform_features(
            raw.reshape(1, -1)
        )
        with self._inference_lock:
            if model == "quantum":
                internal_scope = "full_ensemble" if allow_slow else "fast_vqc"
                explanation = quantum_explainability.explain(
                    quantum_features[0],
                    scope=internal_scope,
                    allow_slow=allow_slow,
                )
                public_scope = "full_ensemble" if allow_slow else "vqc_fast"
            elif model == "classical_full_feature":
                if allow_slow:
                    raise ValueError("allow_slow applies only to the quantum ensemble")
                explanation = bundle.classical_explainability[model].explain(full[0])
                public_scope = model
            elif model == "classical_same_4_feature":
                if allow_slow:
                    raise ValueError("allow_slow applies only to the quantum ensemble")
                explanation = bundle.classical_explainability[model].explain(selected[0])
                public_scope = model
            else:
                raise ValueError("unknown explanation model")

        shap_by_name = {
            item.feature_name: item for item in explanation.shap.attributions
        }
        lime_by_name = {
            item.feature_name: item for item in explanation.lime.attributions
        }
        feature_names = list(explanation.shap.feature_names)
        def public_direction(direction: str) -> str:
            if disease_id == "breast_cancer":
                return direction
            return {
                "toward_benign": "toward_negative_screening",
                "toward_malignant": "toward_positive_screening",
                "neutral": "neutral",
            }[direction]

        top_features = []
        for shap_item in explanation.shap.top_features:
            lime_item = lime_by_name[shap_item.feature_name]
            top_features.append(
                {
                    "feature_name": shap_item.feature_name,
                    "feature_value": shap_item.feature_value,
                    "shap_value": shap_item.attribution,
                    "lime_value": lime_item.attribution,
                    "direction": public_direction(shap_item.direction),
                }
            )
        return {
            "disease_id": disease_id,
            "scope": public_scope,
            "feature_names": feature_names,
            "shap_values": [
                shap_by_name[name].attribution for name in feature_names
            ],
            "lime_values": [
                lime_by_name[name].attribution for name in feature_names
            ],
            "directions": [
                public_direction(shap_by_name[name].direction) for name in feature_names
            ],
            "top_features": top_features,
            "expected_timing": explanation.expected_timing,
            "elapsed_seconds": explanation.elapsed_seconds,
        }

    def ingest_payload(
        self, record: dict[str, float], *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]:
        """Order a named clinical record and flag values outside benchmark ranges."""

        bundle = self._require_loaded(disease_id)
        data = bundle.data
        expected_names = [str(name) for name in data.feature_names]
        supplied = set(record)
        expected = set(expected_names)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            detail = []
            if missing:
                detail.append(f"missing: {', '.join(missing)}")
            if extra:
                detail.append(f"unexpected: {', '.join(extra)}")
            raise ValueError(
                f"record must contain the exact {len(expected_names)}-feature schema ("
                + "; ".join(detail)
                + ")"
            )
        ordered = np.asarray([record[name] for name in expected_names], dtype=float)
        if not np.isfinite(ordered).all():
            raise ValueError("record values must be finite")
        self._validated_raw_row(bundle, ordered.tolist())
        minimum = bundle.raw_features.min(axis=0)
        maximum = bundle.raw_features.max(axis=0)
        warnings = [
            {
                "feature_name": expected_names[index],
                "value": float(value),
                "observed_min": float(minimum[index]),
                "observed_max": float(maximum[index]),
            }
            for index, value in enumerate(ordered)
            if value < minimum[index] or value > maximum[index]
        ]
        return {
            "disease_id": disease_id,
            "feature_names": expected_names,
            "features": ordered.tolist(),
            "warnings": warnings,
        }

    def baselines_payload(
        self, *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]:
        bundle = self._require_loaded(disease_id)
        data, classical = bundle.data, bundle.classical
        configurations: dict[str, Any] = {}
        for configuration in FEATURE_CONFIGURATIONS:
            model_payload = {}
            for model_name in MODEL_NAMES:
                metrics = classical[configuration][model_name].metrics
                result = classical[configuration][model_name]
                model_payload[model_name] = {
                    **asdict(metrics),
                    "fit_seconds": result.fit_seconds,
                    "predict_seconds": result.predict_seconds,
                }
            configurations[configuration] = {
                "feature_count": (
                    int(data.feature_names.size)
                    if configuration == "full_feature"
                    else 4
                ),
                "runtime_seed": self.seed,
                "models": model_payload,
            }
        return {
            "disease_id": disease_id,
            "positive_class": bundle.definition.positive_class_name,
            "runtime_split_note": (
                "The model rows are this process's held-out seed split; judge-facing "
                "claims use the verified three-seed ranges in /metrics."
            ),
            "configurations": configurations,
        }

    def metrics_payload(
        self, *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]:
        bundle = self._require_loaded(disease_id)
        same4 = {
            name: {
                "mean_accuracy": accuracy,
                "source": "decisions.md D-20",
            }
            for name, accuracy in CARRIED_SAME4_CLASSICAL_MEAN_ACCURACIES.items()
        }
        evaluation_root = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation"

        def load_evidence(name: str) -> Any:
            path = evaluation_root / name
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

        live_quantum_metrics = evaluate_classification(
            bundle.data.y_test,
            bundle.quantum.predictions,
            malignant_scores=1.0 - bundle.quantum.positive_probabilities,
            positive_class_name=bundle.definition.positive_class_name,
        )
        return {
            "disease_id": disease_id,
            "dataset_name": bundle.definition.dataset_name,
            "quantum": (
                VERIFIED_PHASE2_QUANTUM
                if disease_id == DEFAULT_DISEASE_ID
                else {
                    "source": "live identified runtime split",
                    "seed": self.seed,
                    "vqc_epochs": self.vqc_epochs,
                    "training_pool_size": bundle.quantum.training_pool_size,
                    "ensemble": asdict(live_quantum_metrics),
                    "finding": (
                        "Second-disease scalability demonstration; this single "
                        "runtime split is not a replacement for repeated or "
                        "external clinical validation."
                    ),
                }
            ),
            "classical": {
                "same_4_feature_three_seed_means": (
                    same4 if disease_id == DEFAULT_DISEASE_ID else None
                ),
                "live_runtime": {
                    configuration: {
                        name: asdict(result.metrics)
                        for name, result in models.items()
                    }
                    for configuration, models in bundle.classical.items()
                },
                "source": (
                    "decisions.md D-20 and architecture.md section 3.7"
                    if disease_id == DEFAULT_DISEASE_ID
                    else "live identified runtime split"
                ),
            },
            "generalization": {
                "classical_three_seed": (
                    load_evidence("classical_three_seed_metrics.json")
                    if disease_id == DEFAULT_DISEASE_ID
                    else None
                ),
                "five_fold_cross_validation": (
                    load_evidence("five_fold_cross_validation.json")
                    if disease_id == DEFAULT_DISEASE_ID
                    else None
                ),
                "early_diabetes_three_seed": (
                    load_evidence("early_diabetes_three_seed.json")
                    if disease_id == "early_diabetes"
                    else None
                ),
            },
            "positioning": (
                "No claim of quantum superiority. Breast oncology has retained "
                "three-seed and five-fold evidence; early diabetes demonstrates "
                "the same pluggable workflow and is reported as a separate module."
            ),
        }

    def report_payload(
        self, features: list[float], *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]:
        """Generate a reproducible, local, non-diagnostic evidence report."""

        bundle = self._require_loaded(disease_id)
        prediction = self.predict_payload(features, disease_id=disease_id)
        raw = self._validated_raw_row(bundle, features)
        selected_indices = np.flatnonzero(bundle.data.selector.get_support())
        selected_measurements = [
            {
                "feature_name": str(bundle.data.feature_names[index]),
                "value": float(raw[index]),
            }
            for index in selected_indices
        ]
        quantum = prediction["quantum"]
        classical = prediction["classical"]["full_feature"]
        matched = prediction["classical"]["same_4_feature"]
        agreement_sentence = (
            "The quantum and full-feature classical systems agree on the predicted class."
            if prediction["agreement"]["agrees"]
            else (
                "The quantum and full-feature classical systems disagree. This "
                "uncertainty is preserved and must not be hidden."
            )
        )
        summary = (
            f"For this {bundle.definition.short_title.lower()} benchmark record, "
            f"the six-model quantum ensemble returned {quantum['label']} at "
            f"{quantum['confidence'] * 100:.1f}% model confidence. The full-feature "
            f"classical baseline returned {classical['label']} at "
            f"{classical['confidence'] * 100:.1f}%, while the matched four-feature "
            f"classical check returned {matched['label']} at "
            f"{matched['confidence'] * 100:.1f}%. {agreement_sentence}"
        )
        classical_metrics = bundle.classical["full_feature"][
            "logistic_regression"
        ].metrics
        quantum_metrics = evaluate_classification(
            bundle.data.y_test,
            bundle.quantum.predictions,
            malignant_scores=1.0 - bundle.quantum.positive_probabilities,
            positive_class_name=bundle.definition.positive_class_name,
        )
        fairness_audit: dict[str, Any] | None = None
        module_limitations: list[str] = []
        if disease_id == "early_diabetes":
            group_values = bundle.raw_features[bundle.data.test_indices, 1]

            def group_rows(predictions: np.ndarray) -> dict[str, Any]:
                rows: dict[str, Any] = {}
                for numeric, label in ((0.0, "female"), (1.0, "male")):
                    mask = group_values == numeric
                    y_true = bundle.data.y_test[mask]
                    y_pred = predictions[mask]
                    present = y_true == 0
                    absent = y_true == 1
                    rows[label] = {
                        "n": int(mask.sum()),
                        "accuracy": float(np.mean(y_true == y_pred)),
                        "condition_sensitivity": (
                            float(np.mean(y_pred[present] == 0))
                            if np.any(present)
                            else None
                        ),
                        "specificity": (
                            float(np.mean(y_pred[absent] == 1))
                            if np.any(absent)
                            else None
                        ),
                    }
                return rows

            fairness_audit = {
                "attribute": "dataset-recorded binary sex/gender field",
                "quantum_ensemble": group_rows(bundle.quantum.predictions),
                "classical_full_feature_logistic_regression": group_rows(
                    bundle.classical["full_feature"][
                        "logistic_regression"
                    ].predictions
                ),
                "interpretation": (
                    "Small held-out subgroup measurements surface possible gaps; "
                    "they are not proof of fairness."
                ),
            }
            module_limitations.extend(
                [
                    "The diabetes data came from one hospital questionnaire study in Bangladesh and has not been validated for India's population.",
                    "The source records a binary sex/gender field and may not represent all patients.",
                    "Strong symptom predictors can inflate apparent accuracy; this is not proof of pre-symptomatic detection.",
                ]
            )
        return {
            "report_id": f"QTRACE-{uuid4().hex[:12].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disease_id": disease_id,
            "title": bundle.definition.title,
            "dataset": {
                "name": bundle.definition.dataset_name,
                "source": bundle.definition.dataset_source,
                "license": bundle.definition.dataset_license,
            },
            "configuration": {
                "configuration_id": self.runtime_manifest["configuration_id"],
                "seed": self.seed,
                "vqc_epochs": self.vqc_epochs,
                "quantum_training_limit": self.training_limit,
                "quantum_device": self.quantum_device,
            },
            "prediction": prediction,
            "selected_measurements": selected_measurements,
            "ai_evidence_summary": summary,
            "held_out_evidence": {
                "positive_class": bundle.definition.positive_class_name,
                "quantum_ensemble": asdict(quantum_metrics),
                "classical_full_feature_logistic_regression": asdict(classical_metrics),
                "scope": (
                    "Measurements from this identified runtime split; retained "
                    "multi-seed/fold evidence is available separately for WBCD."
                ),
            },
            "fairness_audit": fairness_audit,
            "limitations": [
                "Research prototype only; not a medical device or diagnosis.",
                "Model confidence is not a calibrated personal disease risk.",
                "Results come from public benchmark data, not prospective clinical validation.",
                "A qualified clinician must interpret real symptoms and measurements.",
                "No patient record or report is stored by the server.",
                *module_limitations,
            ],
        }


__all__ = ["ModelRuntime", "RuntimeNotReady", "VERIFIED_PHASE2_QUANTUM"]
