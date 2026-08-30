"""Regression tests for the variational quantum classifier."""

from __future__ import annotations

import numpy as numpy

from backend.quantum.vqc_circuits import (
    DEFAULT_N_LAYERS,
    DEFAULT_N_QUBITS,
    train_vqc,
    weight_indices_used,
)


def test_cost_decreases_during_training() -> None:
    """Guard against mixing a Torch QNode with PennyLane's optimizer (D-02)."""
    features = numpy.array(
        [
            [0.10, 0.20, 0.15, 0.05],
            [0.25, 0.10, 0.30, 0.20],
            [0.45, 0.35, 0.40, 0.30],
            [2.55, 2.70, 2.60, 2.75],
            [2.80, 2.65, 2.90, 2.70],
            [3.00, 2.90, 2.75, 3.05],
        ],
        dtype=float,
    )
    labels_pm1 = numpy.array([1, 1, 1, -1, -1, -1], dtype=float)

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
