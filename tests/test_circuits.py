"""Regression tests for the variational quantum classifier."""

from __future__ import annotations

import numpy as numpy
import pennylane as qml
import pytest

from backend.data.pipeline import prepare_breast_cancer_data
from backend.quantum.vqc_circuits import (
    DEFAULT_N_LAYERS,
    DEFAULT_N_QUBITS,
    VQC_VARIANTS,
    make_vqc,
    prepare_vqc_features,
    train_vqc,
    weight_indices_used,
)


EXPECTED_VARIANTS = {
    "vqc_4q_3l_cnot": {
        "n_qubits": 4,
        "n_layers": 3,
        "entangler": "CNOT",
        "feature_indices": (0, 1, 2, 3),
        "input_order": (0, 1, 2, 3),
    },
    "vqc_3q_2l_cnot": {
        "n_qubits": 3,
        "n_layers": 2,
        "entangler": "CNOT",
        "feature_indices": (0, 2, 3),
        "input_order": (0, 1, 2),
    },
    "vqc_4q_2l_cz": {
        "n_qubits": 4,
        "n_layers": 2,
        "entangler": "CZ",
        "feature_indices": (0, 1, 2, 3),
        "input_order": (0, 1, 2, 3),
    },
    "vqc_4q_3l_cz_reversed": {
        "n_qubits": 4,
        "n_layers": 3,
        "entangler": "CZ",
        "feature_indices": (0, 1, 2, 3),
        "input_order": (3, 2, 1, 0),
    },
}


def _operation_names(circuit: qml.QNode, n_qubits: int, n_layers: int) -> list[str]:
    inputs = numpy.linspace(0.1, 0.4, n_qubits)
    weights = numpy.zeros(n_qubits * n_layers * 2)
    tape = qml.workflow.construct_tape(circuit)(inputs, weights)
    return [operation.name for operation in tape.operations]


def test_vqc_variant_registry_matches_architecture() -> None:
    """The four Phase 2 variants must not drift from architecture section 3.6."""
    assert set(VQC_VARIANTS) == set(EXPECTED_VARIANTS)

    for name, expected in EXPECTED_VARIANTS.items():
        config = VQC_VARIANTS[name]
        assert config.n_qubits == expected["n_qubits"]
        assert config.n_layers == expected["n_layers"]
        assert config.entangler == expected["entangler"]
        assert config.feature_indices == expected["feature_indices"]
        assert config.input_order == expected["input_order"]


def test_cnot_and_cz_variants_use_different_entanglers() -> None:
    cnot = make_vqc(n_qubits=4, n_layers=2, entangler="CNOT")
    cz = make_vqc(n_qubits=4, n_layers=2, entangler="CZ")

    cnot_names = _operation_names(cnot, n_qubits=4, n_layers=2)
    cz_names = _operation_names(cz, n_qubits=4, n_layers=2)

    assert cnot_names.count("CNOT") == 2 * (4 - 1)
    assert "CZ" not in cnot_names
    assert cz_names.count("CZ") == 2 * (4 - 1)
    assert "CNOT" not in cz_names


def test_reversed_variant_reverses_data_encoding_on_the_tape() -> None:
    circuit = make_vqc(
        n_qubits=4,
        n_layers=3,
        entangler="CZ",
        input_order=(3, 2, 1, 0),
    )
    inputs = numpy.array([0.1, 0.2, 0.3, 0.4])
    weights = numpy.zeros(4 * 3 * 2)
    tape = qml.workflow.construct_tape(circuit)(inputs, weights)

    operations_per_layer = 4 + (2 * 4) + (4 - 1)
    for layer in range(3):
        layer_start = layer * operations_per_layer
        layer_data = tape.operations[layer_start : layer_start + 4]
        assert [operation.name for operation in layer_data] == ["RY"] * 4
        assert [float(operation.parameters[0]) for operation in layer_data] == (
            pytest.approx([0.4, 0.3, 0.2, 0.1])
        )


def test_three_qubit_variant_selects_exactly_three_features() -> None:
    config = VQC_VARIANTS["vqc_3q_2l_cnot"]
    full_selected_features = numpy.array(
        [[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]]
    )

    prepared = prepare_vqc_features(full_selected_features, config)

    assert prepared.shape == (2, 3)
    numpy.testing.assert_array_equal(prepared, full_selected_features[:, (0, 2, 3)])


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


@pytest.mark.parametrize("config", VQC_VARIANTS.values(), ids=lambda item: item.name)
def test_weight_indices_dont_collide_for_any_variant(config) -> None:
    indices = weight_indices_used(config.n_qubits, config.n_layers)
    expected_count = config.n_qubits * config.n_layers * 2

    assert len(indices) == expected_count
    assert len(set(indices)) == expected_count
    assert indices == tuple(range(expected_count))
