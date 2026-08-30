"""Configuration guards for the expensive Phase 2 training entry point."""

from __future__ import annotations

import pytest

from backend.quantum.train import (
    CARRIED_SAME4_CLASSICAL_MEAN_ACCURACIES,
    DEFAULT_PHASE2_EPOCHS,
    DEFAULT_PHASE2_SEEDS,
    EXPECTED_MEMBER_NAMES,
    MINIMUM_PHASE2_EPOCHS,
    QSVM_VARIANTS,
    run_phase2_benchmark,
)


def test_phase2_registry_contains_exactly_four_vqcs_and_two_qsvms() -> None:
    assert len(EXPECTED_MEMBER_NAMES) == 6
    assert len([name for name in EXPECTED_MEMBER_NAMES if name.startswith("vqc_")]) == 4
    assert QSVM_VARIANTS == {
        "qsvm_angle_4q": "angle",
        "qsvm_amplitude_2q": "amplitude",
    }


def test_phase2_training_refuses_the_20_epoch_regression_budget() -> None:
    assert DEFAULT_PHASE2_EPOCHS >= MINIMUM_PHASE2_EPOCHS == 100

    with pytest.raises(ValueError, match="at least 100 epochs"):
        run_phase2_benchmark(
            seeds=DEFAULT_PHASE2_SEEDS,
            vqc_epochs=20,
            show_progress=False,
        )


def test_phase2_uses_required_three_benchmark_seeds() -> None:
    assert DEFAULT_PHASE2_SEEDS == (42, 123, 2026)


def test_same_four_classical_results_are_carried_forward_unchanged() -> None:
    assert CARRIED_SAME4_CLASSICAL_MEAN_ACCURACIES == {
        "logistic_regression": 0.9357,
        "random_forest": 0.9327,
        "xgboost": 0.9327,
        "svm": 0.9386,
    }
