"""Generalization and reproducible-evidence regressions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer

from backend.data.pipeline import prepare_breast_cancer_partition
from backend.evaluation import run_cross_validation, run_repeated_classical_benchmark


def test_explicit_partition_fits_preprocessing_on_fold_training_rows_only() -> None:
    raw = load_breast_cancer()
    validation = np.arange(0, raw.target.size, 5, dtype=np.int64)
    training = np.setdiff1d(np.arange(raw.target.size), validation)

    prepared = prepare_breast_cancer_partition(training, validation)

    np.testing.assert_array_equal(prepared.train_indices, training)
    np.testing.assert_array_equal(prepared.test_indices, validation)
    np.testing.assert_allclose(
        prepared.imputer.statistics_, np.median(raw.data[training], axis=0)
    )
    assert not np.allclose(
        prepared.full_scaler.mean_, raw.data[validation].mean(axis=0)
    )


def test_five_fold_validation_is_stratified_complete_and_leakage_safe() -> None:
    report = run_cross_validation(
        n_splits=5,
        random_state=42,
        include_quantum=False,
        model_names=("logistic_regression",),
    )

    assert report.n_splits == 5
    assert len(report.folds) == 5
    assert sorted(index for fold in report.folds for index in fold.validation_indices) == list(range(569))
    assert all(set(fold.validation_labels) == {0, 1} for fold in report.folds)
    assert all(fold.preprocessing_fit_rows == len(fold.training_indices) for fold in report.folds)
    assert set(report.summaries) == {
        "classical/full_feature/logistic_regression",
        "classical/same_4_feature/logistic_regression",
    }


def test_repeated_classical_benchmark_retains_complete_metrics_for_both_views() -> None:
    report = run_repeated_classical_benchmark(
        seeds=(42, 123, 2026), model_names=("logistic_regression",)
    )

    assert report.seeds == (42, 123, 2026)
    assert len(report.rows) == 6
    for row in report.rows:
        assert row.metrics.positive_class_name == "malignant"
        assert row.metrics.malignant_sensitivity == row.metrics.recall
        assert row.metrics.confusion_matrix
        assert row.metrics.roc_auc is not None
        assert row.fit_seconds >= 0.0
        assert row.predict_seconds >= 0.0
        assert row.timing_scope == "individual_model_fit_and_heldout_prediction"


def test_retained_five_fold_artifact_contains_complete_hybrid_evidence() -> None:
    path = Path("artifacts/evaluation/five_fold_cross_validation.json")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["n_splits"] == 5
    assert report["includes_quantum"] is True
    assert report["vqc_epochs"] == 100
    assert len(report["folds"]) == 5
    for fold in report["folds"]:
        assert set(fold["training_indices"]).isdisjoint(fold["validation_indices"])
        assert fold["preprocessing_fit_rows"] == len(fold["training_indices"])
        assert all(not name.startswith(("PC", "feature_")) for name in fold["selected_feature_names"])
        quantum_rows = [row for row in fold["rows"] if row["paradigm"] == "quantum"]
        assert len(quantum_rows) == 7
        assert all(row["metrics"]["positive_class_name"] == "malignant" for row in quantum_rows)
        assert all(row["timing_scope"] for row in quantum_rows)

    summaries = report["summaries"]
    assert "quantum/same_4_quantum_range/six_model_oob_ensemble" in summaries
    assert "classical/full_feature/logistic_regression" in summaries
    assert "classical/same_4_feature/logistic_regression" in summaries
