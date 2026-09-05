"""Regressions for the second disease module and generated evidence report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from backend.data.diabetes import (
    DATASET_PATH,
    TARGET_NAMES,
    diabetes_prediction_to_label,
    load_early_stage_diabetes,
    prepare_diabetes_data,
)
from backend.main import create_app
from tests.test_api import StubRuntime


def test_bundled_diabetes_dataset_is_the_cited_unchanged_uci_file() -> None:
    digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    features, targets = load_early_stage_diabetes()

    assert digest == "7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137"
    assert features.shape == (520, 16)
    assert set(np.unique(targets)) == {0, 1}
    assert dict(zip(*np.unique(targets, return_counts=True), strict=True)) == {0: 320, 1: 200}


def test_diabetes_label_direction_and_pipeline_are_explicit_and_leakage_safe() -> None:
    prepared = prepare_diabetes_data(random_state=42)

    assert TARGET_NAMES.tolist() == [
        "positive screening signal",
        "negative screening signal",
    ]
    assert diabetes_prediction_to_label(0) == "positive screening signal"
    assert diabetes_prediction_to_label(1) == "negative screening signal"
    assert prepared.X_train_full.shape == (416, 16)
    assert prepared.X_test_full.shape == (104, 16)
    assert prepared.X_train_selected.shape == (416, 4)
    assert prepared.X_train_quantum.shape == (416, 4)
    assert prepared.selected_feature_names.tolist() == [
        "gender",
        "polyuria",
        "polydipsia",
        "partial paresis",
    ]
    assert np.all(prepared.X_train_quantum > 0.0)
    assert np.all(prepared.X_train_quantum < np.pi)
    assert not np.shares_memory(prepared.y, prepared.y_pm1)


class MultiDiseaseStub(StubRuntime):
    def diseases_payload(self) -> dict[str, Any]:
        return {
            "default_disease_id": "breast_cancer",
            "modules": [
                {"disease_id": "breast_cancer", "feature_names": ["x"] * 30},
                {"disease_id": "early_diabetes", "feature_names": ["x"] * 16},
            ],
        }

    def predict_payload(
        self, features: list[float], *, disease_id: str = "breast_cancer"
    ) -> dict[str, Any]:
        if disease_id == "breast_cancer":
            return super().predict_payload(features)
        assert len(features) == 16
        labels = ["positive screening signal", "negative screening signal"]

        def leaf(probability: float, feature_count: int) -> dict[str, Any]:
            predicted = int(probability >= 0.5)
            return {
                "label": labels[predicted],
                "confidence": probability if predicted else 1.0 - probability,
                "class_one_probability": probability,
                "class_probabilities": {
                    labels[0]: 1.0 - probability,
                    labels[1]: probability,
                },
                "model_name": "stub",
                "feature_count": feature_count,
            }

        quantum = leaf(0.21, 4)
        quantum.update(
            {
                "model_count": 6,
                "per_model": [
                    {
                        "name": f"member_{index}",
                        "paradigm": "VQC" if index < 4 else "QSVM",
                        "label": labels[0],
                        "confidence": 0.79,
                        "class_one_probability": 0.21,
                        "class_probabilities": {labels[0]: 0.79, labels[1]: 0.21},
                        "weight": 1.0 / 6.0,
                    }
                    for index in range(6)
                ],
            }
        )
        return {
            "disease_id": disease_id,
            "disease_title": "Early diabetes screening analysis",
            "dataset_name": "UCI Early Stage Diabetes Risk Prediction",
            "class_labels": labels,
            "quantum": quantum,
            "classical": {
                "primary_configuration": "full_feature",
                "full_feature": leaf(0.18, 16),
                "same_4_feature": leaf(0.24, 4),
            },
            "agreement": {"agrees": True, "message": "Models agree."},
        }

    def report_payload(
        self, features: list[float], *, disease_id: str = "breast_cancer"
    ) -> dict[str, Any]:
        prediction = self.predict_payload(features, disease_id=disease_id)
        return {
            "report_id": "QTRACE-TEST",
            "generated_at": "2026-09-04T00:00:00+00:00",
            "disease_id": disease_id,
            "ai_evidence_summary": "Measured outputs only; not a diagnosis.",
            "prediction": prediction,
            "limitations": ["Research use only."],
        }


def test_second_disease_api_keeps_the_full_dual_paradigm_contract() -> None:
    features = [float(index % 2) for index in range(16)]
    with TestClient(create_app(runtime=MultiDiseaseStub())) as client:
        modules = client.get("/diseases")
        prediction = client.post(
            "/predict",
            json={"disease_id": "early_diabetes", "features": features},
        )
        report = client.post(
            "/report",
            json={"disease_id": "early_diabetes", "features": features},
        )

    assert modules.status_code == prediction.status_code == report.status_code == 200
    payload = prediction.json()
    assert payload["disease_id"] == "early_diabetes"
    assert set(payload["classical"]) == {
        "primary_configuration",
        "full_feature",
        "same_4_feature",
    }
    assert len(payload["quantum"]["per_model"]) == 6
    assert {item["paradigm"] for item in payload["quantum"]["per_model"]} == {
        "VQC",
        "QSVM",
    }
    assert "benign_probability" not in payload["quantum"]
    assert set(payload["quantum"]["class_probabilities"]) == set(payload["class_labels"])
    assert report.json()["prediction"]["classical"]["same_4_feature"]
    assert "not a diagnosis" in report.json()["ai_evidence_summary"]


def test_disease_specific_feature_lengths_are_rejected_before_inference() -> None:
    with TestClient(create_app(runtime=MultiDiseaseStub())) as client:
        response = client.post(
            "/predict",
            json={"disease_id": "early_diabetes", "features": [0.0] * 30},
        )
    assert response.status_code == 422


def test_retained_diabetes_evidence_contains_three_real_seed_runs() -> None:
    report = json.loads(
        Path("artifacts/evaluation/early_diabetes_three_seed.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["seeds"] == [42, 123, 2026]
    assert len(report["runs"]) == 3
    assert report["configuration"]["vqc_epochs"] == 100
    assert report["configuration"]["quantum_training_limit"] == 20
    assert all(len(run["members"]) == 6 for run in report["runs"])
    assert all(
        {member["paradigm"] for member in run["members"]} == {"VQC", "QSVM"}
        for run in report["runs"]
    )
    assert report["summaries"]["ensemble_accuracy"]["mean"] == 0.8301282051282052
    assert "not external" in report["qualification"]
