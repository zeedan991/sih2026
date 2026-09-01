"""FastAPI contract regressions for the Phase 4 backend."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.runtime import RuntimeNotReady


FEATURES = [float(index) / 10.0 for index in range(30)]


def _leaf(label: str, benign_probability: float, *, feature_count: int) -> dict[str, Any]:
    predicted_probability = (
        benign_probability if label == "benign" else 1.0 - benign_probability
    )
    return {
        "label": label,
        "confidence": predicted_probability,
        "benign_probability": benign_probability,
        "model_name": "logistic_regression",
        "feature_count": feature_count,
        "held_out_accuracy": 0.94,
    }


class StubRuntime:
    """Small loaded runtime so contract tests never train quantum models."""

    def start_loading(self) -> None:
        return None

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "models_loaded": True,
            "quantum_members": 6,
            "classical_models": 8,
            "selected_features": [
                "mean concave points",
                "worst radius",
                "worst perimeter",
                "worst concave points",
            ],
        }

    def patients_payload(self, limit: int) -> dict[str, Any]:
        return {
            "feature_names": [f"clinical feature {index}" for index in range(30)],
            "selected_feature_names": self.health_payload()["selected_features"],
            "patients": [
                {
                    "id": 7,
                    "name": "WBCD test patient 007",
                    "true_label": "benign",
                    "features": FEATURES,
                    "selected_values": [0.1, 0.2, 0.3, 0.4],
                    "has_disagreement": True,
                }
            ][:limit],
        }

    def predict_payload(self, features: list[float]) -> dict[str, Any]:
        assert features == FEATURES
        return {
            "quantum": {
                "label": "malignant",
                "confidence": 0.71,
                "benign_probability": 0.29,
                "model_name": "six_model_oob_ensemble",
                "feature_count": 4,
                "model_count": 6,
                "per_model": [
                    {
                        "name": "vqc_4q_3l_cnot",
                        "paradigm": "VQC",
                        "label": "malignant",
                        "confidence": 0.72,
                        "benign_probability": 0.28,
                        "weight": 0.2,
                    },
                    {
                        "name": "qsvm_angle_4q",
                        "paradigm": "QSVM",
                        "label": "benign",
                        "confidence": 0.61,
                        "benign_probability": 0.61,
                        "weight": 0.25,
                    },
                    {
                        "name": "vqc_3q_2l_cnot",
                        "paradigm": "VQC",
                        "label": "malignant",
                        "confidence": 0.66,
                        "benign_probability": 0.34,
                        "weight": 0.15,
                    },
                    {
                        "name": "vqc_4q_2l_cz",
                        "paradigm": "VQC",
                        "label": "malignant",
                        "confidence": 0.69,
                        "benign_probability": 0.31,
                        "weight": 0.15,
                    },
                    {
                        "name": "vqc_4q_3l_cz_reversed",
                        "paradigm": "VQC",
                        "label": "malignant",
                        "confidence": 0.74,
                        "benign_probability": 0.26,
                        "weight": 0.15,
                    },
                    {
                        "name": "qsvm_amplitude_2q",
                        "paradigm": "QSVM",
                        "label": "benign",
                        "confidence": 0.55,
                        "benign_probability": 0.55,
                        "weight": 0.10,
                    },
                ],
            },
            "classical": {
                "primary_configuration": "full_feature",
                "full_feature": _leaf("benign", 0.84, feature_count=30),
                "same_4_feature": _leaf("benign", 0.79, feature_count=4),
            },
            "agreement": {
                "agrees": False,
                "message": (
                    "Quantum and classical models disagree; both results are "
                    "shown without hiding the difference."
                ),
            },
        }

    def explain_payload(
        self,
        features: list[float],
        *,
        allow_slow: bool,
    ) -> dict[str, Any]:
        assert features == FEATURES
        scope = "full_ensemble" if allow_slow else "vqc_fast"
        names = self.health_payload()["selected_features"]
        return {
            "scope": scope,
            "feature_names": names,
            "shap_values": [0.12, -0.08, 0.04, -0.02],
            "lime_values": [0.10, -0.07, 0.03, -0.01],
            "directions": [
                "toward_benign",
                "toward_malignant",
                "toward_benign",
                "toward_malignant",
            ],
            "top_features": [
                {
                    "feature_name": names[0],
                    "feature_value": 0.1,
                    "shap_value": 0.12,
                    "lime_value": 0.10,
                    "direction": "toward_benign",
                }
            ],
            "expected_timing": "at least 70 seconds" if allow_slow else "about 7 seconds",
            "elapsed_seconds": 0.01,
        }

    def baselines_payload(self) -> dict[str, Any]:
        return {
            "positive_class": "benign",
            "configurations": {
                "full_feature": {"feature_count": 30, "models": {}},
                "same_4_feature": {"feature_count": 4, "models": {}},
            },
        }

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "quantum": {"ensemble_accuracy": 0.94},
            "classical": {"full_feature_accuracy": 0.98, "same_4_accuracy": 0.94},
            "positioning": "Rigorously benchmarked; no claim of quantum superiority.",
        }


def _client() -> TestClient:
    return TestClient(create_app(runtime=StubRuntime()))


def _assert_fully_populated(value: Any) -> None:
    if isinstance(value, dict):
        assert value
        for child in value.values():
            _assert_fully_populated(child)
    elif isinstance(value, list):
        assert value
        for child in value:
            _assert_fully_populated(child)
    else:
        assert value is not None


def test_health_and_real_patient_catalog_contracts() -> None:
    with _client() as client:
        health = client.get("/health")
        patients = client.get("/patients", params={"limit": 4})

    assert health.status_code == 200
    assert health.json()["models_loaded"] is True
    assert patients.status_code == 200
    assert len(patients.json()["feature_names"]) == 30
    assert len(patients.json()["selected_feature_names"]) == 4
    assert len(patients.json()["patients"][0]["features"]) == 30


def test_predict_always_returns_quantum_and_both_classical_configurations() -> None:
    with _client() as client:
        response = client.post("/predict", json={"features": FEATURES})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"quantum", "classical", "agreement"}
    assert set(payload["classical"]) == {
        "primary_configuration",
        "full_feature",
        "same_4_feature",
    }
    assert payload["quantum"]["model_count"] == 6
    assert len(payload["quantum"]["per_model"]) == 6
    assert {item["paradigm"] for item in payload["quantum"]["per_model"]} == {
        "VQC",
        "QSVM",
    }
    assert payload["classical"]["full_feature"]["feature_count"] == 30
    assert payload["classical"]["same_4_feature"]["feature_count"] == 4
    _assert_fully_populated(payload["quantum"])
    _assert_fully_populated(payload["classical"])


def test_prediction_label_strings_follow_sklearn_direction() -> None:
    with _client() as client:
        payload = client.post("/predict", json={"features": FEATURES}).json()

    assert payload["quantum"]["benign_probability"] < 0.5
    assert payload["quantum"]["label"] == "malignant"
    assert payload["classical"]["full_feature"]["benign_probability"] >= 0.5
    assert payload["classical"]["full_feature"]["label"] == "benign"


def test_explain_is_attribution_only_and_slow_scope_is_explicit() -> None:
    forbidden = {"confidence", "probability", "benign_probability", "prediction_probability"}
    with _client() as client:
        fast = client.post(
            "/explain",
            json={"features": FEATURES, "model": "quantum"},
        )
        deep = client.post(
            "/explain",
            json={"features": FEATURES, "model": "quantum", "allow_slow": True},
        )

    assert fast.status_code == 200
    assert deep.status_code == 200
    assert fast.json()["scope"] == "vqc_fast"
    assert deep.json()["scope"] == "full_ensemble"
    assert len(fast.json()["feature_names"]) == 4

    def keys_recursively(value: Any) -> set[str]:
        if isinstance(value, dict):
            nested = [keys_recursively(child) for child in value.values()]
            return set(value) | set().union(*nested)
        if isinstance(value, list):
            nested = [keys_recursively(child) for child in value]
            return set().union(*nested)
        return set()

    assert forbidden.isdisjoint(keys_recursively(fast.json()))
    assert forbidden.isdisjoint(keys_recursively(deep.json()))


def test_feature_validation_rejects_wrong_shape_and_non_finite_values() -> None:
    with _client() as client:
        too_short = client.post("/predict", json={"features": FEATURES[:-1]})
        non_finite = client.post(
            "/predict",
            content=(
                '{"features": ['
                + ",".join(["NaN"] + ["0.0"] * 29)
                + "]}"
            ),
            headers={"content-type": "application/json"},
        )

    assert too_short.status_code == 422
    assert non_finite.status_code == 422


def test_baselines_and_metrics_are_available() -> None:
    with _client() as client:
        baselines = client.get("/baselines")
        metrics = client.get("/metrics")

    assert baselines.status_code == 200
    assert set(baselines.json()["configurations"]) == {
        "full_feature",
        "same_4_feature",
    }
    assert metrics.status_code == 200
    assert "no claim of quantum superiority" in metrics.json()["positioning"].lower()


def test_inference_returns_service_unavailable_while_models_load() -> None:
    class LoadingRuntime(StubRuntime):
        def health_payload(self) -> dict[str, Any]:
            payload = super().health_payload()
            payload.update(
                {
                    "status": "loading",
                    "models_loaded": False,
                    "quantum_members": 0,
                    "classical_models": 0,
                    "selected_features": [],
                }
            )
            return payload

        def predict_payload(self, features: list[float]) -> dict[str, Any]:
            raise RuntimeNotReady("models are still loading")

    with TestClient(create_app(runtime=LoadingRuntime())) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json={"features": FEATURES})

    assert health.status_code == 200
    assert health.json()["status"] == "loading"
    assert prediction.status_code == 503
    assert prediction.json()["detail"] == "models are still loading"
