"""FastAPI contract regressions for the Phase 4 backend."""

from __future__ import annotations

from typing import Any
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import json

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
            "runtime_configuration": {
                "configuration_id": "qtrace-multidisease-s42-e100-q20-v1",
                "manifest": "artifacts/models/runtime_manifest.json",
                "loading_mode": "deterministic_refit_from_manifest",
                "quantum_device": "lightning.qubit",
                "seed": 42, "vqc_epochs": 100,
                "quantum_training_limit": 20, "classical_training_rows": 455,
            },
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

    def ingest_payload(self, record: dict[str, float]) -> dict[str, Any]:
        names = [f"clinical feature {index}" for index in range(30)]
        assert set(record) == set(names)
        return {
            "feature_names": names,
            "features": [record[name] for name in names],
            "warnings": [],
        }

    def explain_payload(
        self,
        features: list[float],
        *,
        model: str,
        allow_slow: bool,
    ) -> dict[str, Any]:
        assert features == FEATURES
        if model == "quantum":
            scope = "full_ensemble" if allow_slow else "vqc_fast"
            names = self.health_payload()["selected_features"]
        else:
            scope = model
            names = ([f"clinical feature {index}" for index in range(30)] if model == "classical_full_feature" else self.health_payload()["selected_features"])
        return {
            "scope": scope,
            "feature_names": names,
            "shap_values": [0.12 if index == 0 else 0.0 for index in range(len(names))],
            "lime_values": [0.10 if index == 0 else 0.0 for index in range(len(names))],
            "directions": ["toward_benign" if index == 0 else "neutral" for index in range(len(names))],
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
            "positive_class": "malignant",
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
    assert health.json()["runtime_configuration"]["quantum_training_limit"] == 20
    assert patients.status_code == 200
    assert len(patients.json()["feature_names"]) == 30
    assert len(patients.json()["selected_feature_names"]) == 4
    assert len(patients.json()["patients"][0]["features"]) == 30


def test_fastapi_serves_the_judge_frontend_from_the_same_origin() -> None:
    with _client() as client:
        page = client.get("/")
        script = client.get("/static/app.js")

    assert page.status_code == 200
    assert "Q-TRACE" in page.text
    assert 'id="quantum-progress"' in page.text
    assert 'id="classical-progress"' in page.text
    assert script.status_code == 200
    assert 'fetchJson("/predict"' in script.text


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


def test_classical_explanations_use_real_feature_names_and_no_second_verdict() -> None:
    forbidden = {"confidence", "probability", "benign_probability", "prediction_probability"}
    with _client() as client:
        full = client.post(
            "/explain",
            json={"features": FEATURES, "model": "classical_full_feature"},
        )
        matched = client.post(
            "/explain",
            json={"features": FEATURES, "model": "classical_same_4_feature"},
        )

    assert full.status_code == matched.status_code == 200
    assert full.json()["scope"] == "classical_full_feature"
    assert matched.json()["scope"] == "classical_same_4_feature"
    assert len(full.json()["feature_names"]) == 30
    assert len(matched.json()["feature_names"]) == 4
    assert all(not name.startswith("feature_") for name in full.json()["feature_names"])
    assert forbidden.isdisjoint(full.json())


def test_named_record_ingestion_orders_features_and_reports_range_warnings() -> None:
    names = [f"clinical feature {index}" for index in range(30)]
    record = {name: FEATURES[index] for index, name in enumerate(names)}
    with _client() as client:
        response = client.post("/ingest", json={"record": record})

    assert response.status_code == 200
    assert response.json()["features"] == FEATURES
    assert response.json()["feature_names"] == names


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


def test_oversized_json_is_rejected_before_model_work() -> None:
    body = " " * 20_000 + json.dumps({"features": FEATURES})
    with _client() as client:
        response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_chunked_body_cannot_bypass_request_size_limit() -> None:
    chunks = iter([b" " * 9_000, b" " * 9_000, json.dumps({"features": FEATURES}).encode()])
    with _client() as client:
        response = client.post("/predict", content=chunks, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_busy_explanation_rejects_new_inference_without_blocking_health() -> None:
    entered, release = Event(), Event()

    class BusyRuntime(StubRuntime):
        def explain_payload(self, features: list[float], *, model: str, allow_slow: bool) -> dict[str, Any]:
            entered.set()
            assert release.wait(timeout=5), "test must release the simulated explanation"
            return super().explain_payload(features, model=model, allow_slow=allow_slow)

    with TestClient(create_app(runtime=BusyRuntime())) as client, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/explain", json={"features": FEATURES})
        try:
            assert entered.wait(timeout=2)
            competing = client.post("/predict", json={"features": FEATURES})
            health = client.get("/health")
            assert competing.status_code == 429
            assert competing.headers["retry-after"] == "2"
            assert health.status_code == 200
        finally:
            release.set()
        assert first.result(timeout=2).status_code == 200
        assert client.post("/predict", json={"features": FEATURES}).status_code == 200


def test_inference_slot_is_released_after_a_runtime_error() -> None:
    class FlakyRuntime(StubRuntime):
        calls = 0

        def predict_payload(self, features: list[float]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeNotReady("models are still loading")
            return super().predict_payload(features)

    with TestClient(create_app(runtime=FlakyRuntime())) as client:
        assert client.post("/predict", json={"features": FEATURES}).status_code == 503
        assert client.post("/predict", json={"features": FEATURES}).status_code == 200


def test_real_runtime_summary_uses_correct_label_strings() -> None:
    from backend.runtime import _prediction_summary

    for probability, label in ((0.0, "malignant"), (0.49, "malignant"), (0.5, "benign"), (1.0, "benign")):
        result = _prediction_summary(probability, model_name="regression", feature_count=4)
        assert result["label"] == label
        assert result["confidence"] == max(probability, 1.0 - probability)


def test_health_reports_actual_runtime_configuration_before_training() -> None:
    from backend.runtime import ModelRuntime

    runtime = ModelRuntime()
    config = runtime.health_payload()["runtime_configuration"]
    assert config["seed"] == runtime.seed
    assert config["vqc_epochs"] == runtime.vqc_epochs
    assert config["quantum_training_limit"] == runtime.training_limit
    assert config["classical_training_rows"] == 0
    assert config["configuration_id"] == "qtrace-multidisease-s42-e100-q20-v1"
    assert config["loading_mode"] == "deterministic_refit_from_manifest"
