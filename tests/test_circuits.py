"""Regression tests for the variational quantum classifier."""

from __future__ import annotations

import numpy as numpy

from backend.data.pipeline import prepare_breast_cancer_data
from backend.quantum.vqc_circuits import (
    DEFAULT_N_LAYERS,
    DEFAULT_N_QUBITS,
    train_vqc,
    weight_indices_used,
)


def test_cost_decreases_during_training() -> None:
    """Guard against mixing a Torch QNode with PennyLane's optimizer (D-02)."""
    prepared = prepare_breast_cancer_data(random_state=42)
    malignant = numpy.flatnonzero(prepared.y_train_pm1 == -1)[:3]
    benign = numpy.flatnonzero(prepared.y_train_pm1 == 1)[:3]
    subset = numpy.concatenate((malignant, benign))
    features = prepared.X_train_quantum[subset]
    labels_pm1 = prepared.y_train_pm1[subset]

    result = train_vqc(
        features,
        labels_pm1,
        seed=7,
        n_epochs=20,
        learning_rate=0.08,
    )

    assert len(result.cost_history) == 21
    assert result.cost_history[-1] < result.cost_history[0]


def test_weight_indices_dont_collide() -> None:
    """Every RY/RZ trainable rotation must consume its own raw weight."""
    indices = weight_indices_used(DEFAULT_N_QUBITS, DEFAULT_N_LAYERS)
    expected_count = DEFAULT_N_QUBITS * DEFAULT_N_LAYERS * 2

    assert len(indices) == expected_count
    assert len(set(indices)) == expected_count
    assert indices == tuple(range(expected_count))
