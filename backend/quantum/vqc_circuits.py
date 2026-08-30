"""Variational quantum classifier circuit and deterministic training helpers.

The implementation deliberately stays within PennyLane's NumPy/Autograd
ecosystem.  In particular, the QNode is never configured with the Torch
interface while being trained by :class:`qml.AdamOptimizer` (decision D-02).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as ordinary_numpy
import pennylane as qml
from pennylane import numpy as np

DEFAULT_N_QUBITS = 4
DEFAULT_N_LAYERS = 3
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_N_EPOCHS = 100
_PROBABILITY_EPSILON = 1e-7
EntanglerName = Literal["CNOT", "CZ"]


def _validate_input_order(
    input_order: Sequence[int] | None,
    n_qubits: int,
) -> tuple[int, ...]:
    """Return a validated feature-to-wire encoding order."""
    if input_order is None:
        return tuple(range(n_qubits))
    order = tuple(int(index) for index in input_order)
    if len(order) != n_qubits or set(order) != set(range(n_qubits)):
        raise ValueError(
            "input_order must be a permutation of "
            f"{tuple(range(n_qubits))}, received {order}"
        )
    return order


def _validate_entangler(entangler: str) -> EntanglerName:
    normalized = entangler.upper()
    if normalized not in {"CNOT", "CZ"}:
        raise ValueError("entangler must be either 'CNOT' or 'CZ'")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class VQCVariantConfig:
    """Immutable specification for one architecture section 3.6 VQC."""

    name: str
    n_qubits: int
    n_layers: int
    entangler: EntanglerName
    feature_indices: tuple[int, ...]
    input_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        normalized_entangler = _validate_entangler(self.entangler)
        if normalized_entangler != self.entangler:
            raise ValueError("entangler names in VQC configs must be uppercase")
        if len(self.feature_indices) != self.n_qubits:
            raise ValueError("feature_indices must contain one index per qubit")
        if len(set(self.feature_indices)) != len(self.feature_indices):
            raise ValueError("feature_indices cannot contain duplicates")
        if any(index < 0 for index in self.feature_indices):
            raise ValueError("feature_indices cannot contain negative indices")
        _validate_input_order(self.input_order, self.n_qubits)


_VQC_VARIANT_CONFIGS = (
    VQCVariantConfig(
        name="vqc_4q_3l_cnot",
        n_qubits=4,
        n_layers=3,
        entangler="CNOT",
        feature_indices=(0, 1, 2, 3),
        input_order=(0, 1, 2, 3),
    ),
    VQCVariantConfig(
        name="vqc_3q_2l_cnot",
        n_qubits=3,
        n_layers=2,
        entangler="CNOT",
        feature_indices=(0, 2, 3),
        input_order=(0, 1, 2),
    ),
    VQCVariantConfig(
        name="vqc_4q_2l_cz",
        n_qubits=4,
        n_layers=2,
        entangler="CZ",
        feature_indices=(0, 1, 2, 3),
        input_order=(0, 1, 2, 3),
    ),
    VQCVariantConfig(
        name="vqc_4q_3l_cz_reversed",
        n_qubits=4,
        n_layers=3,
        entangler="CZ",
        feature_indices=(0, 1, 2, 3),
        input_order=(3, 2, 1, 0),
    ),
)

VQC_VARIANTS: Mapping[str, VQCVariantConfig] = MappingProxyType(
    {config.name: config for config in _VQC_VARIANT_CONFIGS}
)


def prepare_vqc_features(
    features: Sequence[Sequence[float]],
    config: VQCVariantConfig,
) -> ordinary_numpy.ndarray:
    """Select the architecture-declared input columns for one VQC variant.

    The four-feature variants preserve all selected clinical features.  The
    shallow three-qubit variant consumes the three strongest ANOVA features
    from the verified four-feature pipeline, kept in pipeline column order.
    Reversed *encoding* remains a circuit concern represented by
    ``config.input_order`` rather than an accidental second column reversal.
    """
    feature_matrix = ordinary_numpy.asarray(features, dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if feature_matrix.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    if max(config.feature_indices) >= feature_matrix.shape[1]:
        raise ValueError(
            f"Variant {config.name!r} needs feature columns "
            f"{config.feature_indices}, but received {feature_matrix.shape[1]} columns"
        )
    selected = feature_matrix[:, config.feature_indices]
    if not ordinary_numpy.isfinite(selected).all():
        raise ValueError("features must contain only finite values")
    return ordinary_numpy.asarray(selected, dtype=float)


def number_of_weights(n_qubits: int, n_layers: int) -> int:
    """Return the number of raw weights consumed by a VQC."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    return n_qubits * n_layers * 2


def weight_indices_used(n_qubits: int, n_layers: int) -> tuple[int, ...]:
    """List weight indices in circuit-consumption order.

    This mirrors the circuit's running-counter pattern and exists so the
    collision regression can assert that each RY/RZ rotation gets a distinct
    parameter.  A layer-based index formula is intentionally not used.
    """
    number_of_weights(n_qubits, n_layers)  # validate both dimensions
    indices: list[int] = []
    idx = 0
    for _layer in range(n_layers):
        for _wire in range(n_qubits):
            indices.append(idx)
            idx += 1
            indices.append(idx)
            idx += 1
    return tuple(indices)


def remap_weights(raw_weights: np.ndarray) -> np.ndarray:
    """Sigmoid-map unconstrained trainable weights into rotation angles."""
    return 2 * np.pi * (1 / (1 + np.exp(-raw_weights)))


def make_vqc(
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    *,
    entangler: EntanglerName | str = "CNOT",
    input_order: Sequence[int] | None = None,
) -> qml.QNode:
    """Create an independent data-reuploading VQC on a fresh device.

    Inputs are encoded with RY rotations at every variational layer.  Every
    qubit then receives one trainable RY and RZ rotation, followed by a linear
    nearest-neighbour CNOT or CZ chain.  ``input_order`` controls which input
    is encoded on each successive wire.  The returned expectation is in
    ``[-1, 1]``.
    """
    expected_weight_count = number_of_weights(n_qubits, n_layers)
    resolved_entangler = _validate_entangler(entangler)
    resolved_input_order = _validate_input_order(input_order, n_qubits)
    device = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(device, interface="autograd")
    def circuit(inputs: np.ndarray, raw_weights: np.ndarray) -> np.ndarray:
        if len(inputs) != n_qubits:
            raise ValueError(
                f"Expected {n_qubits} input features, received {len(inputs)}"
            )
        if len(raw_weights) != expected_weight_count:
            raise ValueError(
                f"Expected {expected_weight_count} raw weights, "
                f"received {len(raw_weights)}"
            )

        weights = remap_weights(raw_weights)
        idx = 0
        for _layer in range(n_layers):
            # D-11: re-encode the classical inputs at every layer.
            for wire in range(n_qubits):
                qml.RY(inputs[resolved_input_order[wire]], wires=wire)

            for wire in range(n_qubits):
                qml.RY(weights[idx], wires=wire)
                idx += 1
                qml.RZ(weights[idx], wires=wire)
                idx += 1

            for wire in range(n_qubits - 1):
                if resolved_entangler == "CNOT":
                    qml.CNOT(wires=[wire, wire + 1])
                else:
                    qml.CZ(wires=[wire, wire + 1])

        return qml.expval(qml.PauliZ(0))

    return circuit


def initialize_vqc_weights(
    *,
    seed: int,
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    scale: float = 0.1,
) -> np.ndarray:
    """Create deterministic trainable raw weights for one VQC run."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    rng = ordinary_numpy.random.default_rng(seed)
    values = rng.normal(0.0, scale, size=number_of_weights(n_qubits, n_layers))
    return np.array(values, requires_grad=True)


def _as_feature_matrix(
    features: Sequence[Sequence[float]], n_qubits: int
) -> np.ndarray:
    ordinary = ordinary_numpy.asarray(features, dtype=float)
    if ordinary.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if ordinary.shape[1] != n_qubits:
        raise ValueError(
            f"Expected {n_qubits} features per sample, received {ordinary.shape[1]}"
        )
    if ordinary.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    if not ordinary_numpy.isfinite(ordinary).all():
        raise ValueError("features must contain only finite values")
    # Clinical feature values are data, never optimizer parameters.
    return np.array(ordinary, requires_grad=False)


def _as_pm1_labels(labels_pm1: Sequence[float], n_samples: int) -> np.ndarray:
    ordinary = ordinary_numpy.asarray(labels_pm1, dtype=float)
    if ordinary.ndim != 1 or ordinary.shape[0] != n_samples:
        raise ValueError("labels_pm1 must be one-dimensional and match features")
    if not ordinary_numpy.isin(ordinary, (-1.0, 1.0)).all():
        raise ValueError("labels_pm1 must contain only -1 and +1")
    return np.array(ordinary, requires_grad=False)


def vqc_raw_outputs(
    circuit: qml.QNode,
    features: np.ndarray,
    raw_weights: np.ndarray,
) -> np.ndarray:
    """Evaluate Pauli-Z expectations for a feature matrix."""
    return np.stack([circuit(sample, raw_weights) for sample in features])


def vqc_positive_probabilities(
    circuit: qml.QNode,
    features: np.ndarray,
    raw_weights: np.ndarray,
) -> np.ndarray:
    """Return ``P(y=+1)`` (equivalently ``P(y=1)``) for each sample."""
    return (vqc_raw_outputs(circuit, features, raw_weights) + 1.0) / 2.0


def binary_cross_entropy(
    circuit: qml.QNode,
    raw_weights: np.ndarray,
    features: np.ndarray,
    labels_pm1: np.ndarray,
) -> np.ndarray:
    """Compute corrected BCE while retaining ``{-1,+1}`` model labels.

    D-03 requires the third label representation, ``{0,1}``, to exist only
    inside this loss calculation.  The caller's ``labels_pm1`` is never
    overwritten or mutated.
    """
    labels_01 = (labels_pm1 + 1.0) / 2.0
    probabilities = vqc_positive_probabilities(circuit, features, raw_weights)
    probabilities = np.clip(
        probabilities,
        _PROBABILITY_EPSILON,
        1.0 - _PROBABILITY_EPSILON,
    )
    return -np.mean(
        labels_01 * np.log(probabilities)
        + (1.0 - labels_01) * np.log(1.0 - probabilities)
    )


@dataclass(frozen=True)
class VQCTrainingResult:
    """Artifacts from a deterministic VQC training run."""

    circuit: qml.QNode
    raw_weights: np.ndarray
    cost_history: tuple[float, ...]
    seed: int
    n_qubits: int
    n_layers: int
    entangler: EntanglerName
    input_order: tuple[int, ...]


def train_vqc(
    features: Sequence[Sequence[float]],
    labels_pm1: Sequence[float],
    *,
    seed: int = 0,
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    entangler: EntanglerName | str = "CNOT",
    input_order: Sequence[int] | None = None,
    n_epochs: int = DEFAULT_N_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    initial_weights: Sequence[float] | None = None,
) -> VQCTrainingResult:
    """Train one VQC with full-batch Adam and return every epoch's cost.

    ``cost_history[0]`` is the untrained cost; each subsequent value is the
    cost after one optimizer update, so a 20-epoch run returns 21 values.
    """
    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    feature_array = _as_feature_matrix(features, n_qubits)
    label_array = _as_pm1_labels(labels_pm1, feature_array.shape[0])
    resolved_entangler = _validate_entangler(entangler)
    resolved_input_order = _validate_input_order(input_order, n_qubits)
    circuit = make_vqc(
        n_qubits,
        n_layers,
        entangler=resolved_entangler,
        input_order=resolved_input_order,
    )

    if initial_weights is None:
        raw_weights = initialize_vqc_weights(
            seed=seed,
            n_qubits=n_qubits,
            n_layers=n_layers,
        )
    else:
        ordinary_weights = ordinary_numpy.asarray(initial_weights, dtype=float)
        expected_shape = (number_of_weights(n_qubits, n_layers),)
        if ordinary_weights.shape != expected_shape:
            raise ValueError(
                f"initial_weights must have shape {expected_shape}, "
                f"received {ordinary_weights.shape}"
            )
        raw_weights = np.array(ordinary_weights, requires_grad=True)

    def objective(weights: np.ndarray) -> np.ndarray:
        return binary_cross_entropy(
            circuit,
            weights,
            feature_array,
            label_array,
        )

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    cost_history = [float(objective(raw_weights))]
    for _epoch in range(n_epochs):
        raw_weights = optimizer.step(objective, raw_weights)
        cost_history.append(float(objective(raw_weights)))

    return VQCTrainingResult(
        circuit=circuit,
        raw_weights=raw_weights,
        cost_history=tuple(cost_history),
        seed=seed,
        n_qubits=n_qubits,
        n_layers=n_layers,
        entangler=resolved_entangler,
        input_order=resolved_input_order,
    )


def predict_vqc_probabilities(
    circuit: qml.QNode,
    raw_weights: Sequence[float],
    features: Sequence[Sequence[float]],
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
) -> ordinary_numpy.ndarray:
    """Return benign/positive-class probabilities as an ordinary array."""
    feature_array = _as_feature_matrix(features, n_qubits)
    weight_array = np.array(raw_weights, requires_grad=False)
    probabilities = vqc_positive_probabilities(circuit, feature_array, weight_array)
    return ordinary_numpy.asarray(probabilities, dtype=float)


def predict_vqc(
    circuit: qml.QNode,
    raw_weights: Sequence[float],
    features: Sequence[Sequence[float]],
    *,
    n_qubits: int = DEFAULT_N_QUBITS,
    threshold: float = 0.5,
) -> ordinary_numpy.ndarray:
    """Return original dataset labels ``{0,1}`` (0 malignant, 1 benign)."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    probabilities = predict_vqc_probabilities(
        circuit,
        raw_weights,
        features,
        n_qubits=n_qubits,
    )
    return (probabilities >= threshold).astype(int)
