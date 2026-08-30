"""Train and benchmark the complete Roadmap Phase 2 quantum ensemble.

This module owns orchestration only. Circuit definitions live in
``vqc_circuits`` and fidelity-kernel SVMs live in ``qsvm_kernels``. Every
member receives its own bootstrap sample; its inverse-MSE weight comes from
that member's out-of-bag rows, never from the held-out test set.

Run the required three-seed benchmark from the repository root with::

    python -m backend.quantum.train
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from statistics import mean
from typing import Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from backend.data.pipeline import PreparedBreastCancerData, prepare_breast_cancer_data
from backend.quantum.ensemble import (
    OOBBootstrap,
    OOBPrediction,
    PairedCorrectnessComparison,
    inverse_oob_mse_weights,
    make_oob_bootstrap,
    paired_correctness_ttest,
    weighted_soft_vote,
)
from backend.quantum.qsvm_kernels import QuantumKernelSVM
from backend.quantum.vqc_circuits import (
    DEFAULT_LEARNING_RATE,
    VQC_VARIANTS,
    VQCTrainingResult,
    VQCVariantConfig,
    predict_vqc_probabilities,
    prepare_vqc_features,
    train_vqc,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Paradigm = Literal["VQC", "QSVM"]

DEFAULT_PHASE2_SEEDS: Final[tuple[int, ...]] = (42, 123, 2026)
DEFAULT_PHASE2_EPOCHS: Final[int] = 100
MINIMUM_PHASE2_EPOCHS: Final[int] = 100
QSVM_VARIANTS: Final[Mapping[str, Literal["angle", "amplitude"]]] = {
    "qsvm_angle_4q": "angle",
    "qsvm_amplitude_2q": "amplitude",
}
EXPECTED_MEMBER_NAMES: Final[tuple[str, ...]] = (
    *VQC_VARIANTS.keys(),
    *QSVM_VARIANTS.keys(),
)

# D-20 explicitly says to carry these Phase 1 results forward rather than
# rerunning the same classical benchmark during the expensive Phase 2 run.
CARRIED_SAME4_CLASSICAL_MEAN_ACCURACIES: Final[Mapping[str, float]] = {
    "logistic_regression": 0.9357,
    "random_forest": 0.9327,
    "xgboost": 0.9327,
    "svm": 0.9386,
}


@dataclass(frozen=True, slots=True)
class MemberEvaluation:
    """Fitted artifact and held-out/OOB measurements for one quantum model."""

    name: str
    paradigm: Paradigm
    configuration: str
    oob_size: int
    oob_mse: float
    weight: float
    accuracy: float
    positive_probabilities: FloatArray
    predictions: IntArray
    artifact: VQCTrainingResult | QuantumKernelSVM
    vqc_config: VQCVariantConfig | None = None

    def predict_benign_proba(self, quantum_features: FloatArray) -> FloatArray:
        """Predict class-one/benign probabilities for new quantum-range rows."""

        if self.paradigm == "VQC":
            if self.vqc_config is None or not isinstance(
                self.artifact, VQCTrainingResult
            ):
                raise RuntimeError("VQC member is missing its fitted configuration")
            features = prepare_vqc_features(quantum_features, self.vqc_config)
            return predict_vqc_probabilities(
                self.artifact.circuit,
                self.artifact.raw_weights,
                features,
                n_qubits=self.vqc_config.n_qubits,
            )

        if not isinstance(self.artifact, QuantumKernelSVM):
            raise RuntimeError("QSVM member is missing its fitted estimator")
        return self.artifact.predict_benign_proba(quantum_features)


@dataclass(frozen=True, slots=True)
class SeedEnsembleResult:
    """All six member results and the ensemble comparison for one data split."""

    seed: int
    members: tuple[MemberEvaluation, ...]
    positive_probabilities: FloatArray
    predictions: IntArray
    accuracy: float
    best_single_name: str
    best_single_accuracy: float
    paired_comparison: PairedCorrectnessComparison
    training_pool_size: int
    elapsed_seconds: float

    def predict_benign_proba(self, quantum_features: FloatArray) -> FloatArray:
        """Run all fitted members and apply their frozen OOB weights."""

        member_probabilities = {
            member.name: member.predict_benign_proba(quantum_features)
            for member in self.members
        }
        weights = {member.name: member.weight for member in self.members}
        return weighted_soft_vote(member_probabilities, weights)


@dataclass(frozen=True, slots=True)
class Phase2BenchmarkReport:
    """Measured results across the required independent random seeds."""

    seed_results: tuple[SeedEnsembleResult, ...]
    seeds: tuple[int, ...]
    vqc_epochs: int
    learning_rate: float
    training_sample_limit: int | None
    total_seconds: float


@dataclass(slots=True)
class _PendingMember:
    name: str
    paradigm: Paradigm
    configuration: str
    oob_size: int
    oob_prediction: OOBPrediction
    positive_probabilities: FloatArray
    predictions: IntArray
    accuracy: float
    artifact: VQCTrainingResult | QuantumKernelSVM
    vqc_config: VQCVariantConfig | None = None


def _validate_phase2_epochs(vqc_epochs: int) -> None:
    if vqc_epochs < MINIMUM_PHASE2_EPOCHS:
        raise ValueError(
            f"Phase 2 VQCs require at least {MINIMUM_PHASE2_EPOCHS} epochs; "
            "20 epochs is only the D-02 regression budget"
        )


def _training_pool_indices(
    labels: IntArray,
    *,
    seed: int,
    sample_limit: int | None,
) -> IntArray:
    indices = np.arange(labels.size, dtype=np.int64)
    if sample_limit is None or sample_limit == labels.size:
        return indices
    if sample_limit < 20:
        raise ValueError("training_sample_limit must be at least 20")
    if sample_limit > labels.size:
        raise ValueError("training_sample_limit cannot exceed the training split")
    selected, _unused = train_test_split(
        indices,
        train_size=sample_limit,
        random_state=seed,
        stratify=labels,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))


def _binary_oob_bootstrap(labels: IntArray, *, seed: int) -> OOBBootstrap:
    """Draw until both the bootstrap and OOB portions contain both labels."""

    for offset in range(100):
        split = make_oob_bootstrap(n_samples=labels.size, seed=seed + offset)
        if (
            np.unique(labels[split.bootstrap_indices]).size == 2
            and np.unique(labels[split.oob_indices]).size == 2
        ):
            return split
    raise RuntimeError("could not draw a class-complete bootstrap/OOB split")


def _member_seed(seed: int, member_index: int, *, stream: int) -> int:
    generated = np.random.SeedSequence([seed, member_index, stream]).generate_state(1)
    return int(generated[0])


def _accuracy(targets: IntArray, predictions: IntArray) -> float:
    return float(np.mean(np.asarray(targets) == np.asarray(predictions)))


def train_quantum_ensemble(
    data: PreparedBreastCancerData,
    *,
    seed: int,
    vqc_epochs: int = DEFAULT_PHASE2_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    training_sample_limit: int | None = None,
    show_progress: bool = True,
) -> SeedEnsembleResult:
    """Fit four VQCs and two QSVMs, then derive OOB weights and test metrics."""

    _validate_phase2_epochs(vqc_epochs)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    started = time.perf_counter()
    pool_indices = _training_pool_indices(
        data.y_train,
        seed=seed,
        sample_limit=training_sample_limit,
    )
    pool_features = data.X_train_quantum[pool_indices]
    pool_y = data.y_train[pool_indices]
    pool_y_pm1 = data.y_train_pm1[pool_indices]
    pending: list[_PendingMember] = []

    for member_index, config in enumerate(VQC_VARIANTS.values()):
        member_started = time.perf_counter()
        split = _binary_oob_bootstrap(
            pool_y,
            seed=_member_seed(seed, member_index, stream=1),
        )
        variant_features = prepare_vqc_features(pool_features, config)
        fitted = train_vqc(
            variant_features[split.bootstrap_indices],
            pool_y_pm1[split.bootstrap_indices],
            seed=_member_seed(seed, member_index, stream=2),
            n_qubits=config.n_qubits,
            n_layers=config.n_layers,
            entangler=config.entangler,
            input_order=config.input_order,
            n_epochs=vqc_epochs,
            learning_rate=learning_rate,
        )
        oob_probabilities = predict_vqc_probabilities(
            fitted.circuit,
            fitted.raw_weights,
            variant_features[split.oob_indices],
            n_qubits=config.n_qubits,
        )
        test_features = prepare_vqc_features(data.X_test_quantum, config)
        test_probabilities = predict_vqc_probabilities(
            fitted.circuit,
            fitted.raw_weights,
            test_features,
            n_qubits=config.n_qubits,
        )
        test_predictions = (test_probabilities >= 0.5).astype(np.int64)
        pending.append(
            _PendingMember(
                name=config.name,
                paradigm="VQC",
                configuration=(
                    f"{config.n_qubits}q/{config.n_layers}l/{config.entangler}/"
                    f"features={config.feature_indices}/order={config.input_order}"
                ),
                oob_size=int(split.oob_indices.size),
                oob_prediction=OOBPrediction(
                    targets=pool_y[split.oob_indices],
                    positive_probabilities=oob_probabilities,
                ),
                positive_probabilities=test_probabilities,
                predictions=test_predictions,
                accuracy=_accuracy(data.y_test, test_predictions),
                artifact=fitted,
                vqc_config=config,
            )
        )
        if show_progress:
            print(
                f"  VQC  {config.name}: accuracy={pending[-1].accuracy:.4f}, "
                f"cost={fitted.cost_history[0]:.6f}->"
                f"{fitted.cost_history[-1]:.6f}, "
                f"{time.perf_counter() - member_started:.1f}s",
                flush=True,
            )

    qsvm_offset = len(VQC_VARIANTS)
    for variant_offset, (name, embedding) in enumerate(QSVM_VARIANTS.items()):
        member_index = qsvm_offset + variant_offset
        member_started = time.perf_counter()
        split = _binary_oob_bootstrap(
            pool_y,
            seed=_member_seed(seed, member_index, stream=1),
        )
        fitted = QuantumKernelSVM(
            embedding=embedding,
            random_state=_member_seed(seed, member_index, stream=2),
        ).fit(pool_features[split.bootstrap_indices], pool_y[split.bootstrap_indices])
        oob_probabilities = fitted.predict_benign_proba(
            pool_features[split.oob_indices]
        )
        test_probabilities = fitted.predict_benign_proba(data.X_test_quantum)
        test_predictions = (test_probabilities >= 0.5).astype(np.int64)
        pending.append(
            _PendingMember(
                name=name,
                paradigm="QSVM",
                configuration=(
                    f"{fitted.n_qubits}q/{embedding}-embedding/"
                    "SVC(precomputed)"
                ),
                oob_size=int(split.oob_indices.size),
                oob_prediction=OOBPrediction(
                    targets=pool_y[split.oob_indices],
                    positive_probabilities=oob_probabilities,
                ),
                positive_probabilities=test_probabilities,
                predictions=test_predictions,
                accuracy=_accuracy(data.y_test, test_predictions),
                artifact=fitted,
            )
        )
        if show_progress:
            print(
                f"  QSVM {name}: accuracy={pending[-1].accuracy:.4f}, "
                f"{time.perf_counter() - member_started:.1f}s",
                flush=True,
            )

    if tuple(member.name for member in pending) != EXPECTED_MEMBER_NAMES:
        raise RuntimeError("the Phase 2 ensemble must contain the exact six models")

    weighting = inverse_oob_mse_weights(
        {member.name: member.oob_prediction for member in pending}
    )
    members = tuple(
        MemberEvaluation(
            name=member.name,
            paradigm=member.paradigm,
            configuration=member.configuration,
            oob_size=member.oob_size,
            oob_mse=weighting.mse_by_model[member.name],
            weight=weighting.weights[member.name],
            accuracy=member.accuracy,
            positive_probabilities=member.positive_probabilities,
            predictions=member.predictions,
            artifact=member.artifact,
            vqc_config=member.vqc_config,
        )
        for member in pending
    )
    ensemble_probabilities = weighted_soft_vote(
        {member.name: member.positive_probabilities for member in members},
        {member.name: member.weight for member in members},
    )
    ensemble_predictions = (ensemble_probabilities >= 0.5).astype(np.int64)
    ensemble_accuracy = _accuracy(data.y_test, ensemble_predictions)
    best_single = max(members, key=lambda member: member.accuracy)
    comparison = paired_correctness_ttest(
        data.y_test,
        ensemble_predictions,
        best_single.predictions,
    )

    return SeedEnsembleResult(
        seed=seed,
        members=members,
        positive_probabilities=ensemble_probabilities,
        predictions=ensemble_predictions,
        accuracy=ensemble_accuracy,
        best_single_name=best_single.name,
        best_single_accuracy=best_single.accuracy,
        paired_comparison=comparison,
        training_pool_size=int(pool_indices.size),
        elapsed_seconds=time.perf_counter() - started,
    )


def run_phase2_benchmark(
    *,
    seeds: Sequence[int] = DEFAULT_PHASE2_SEEDS,
    vqc_epochs: int = DEFAULT_PHASE2_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    training_sample_limit: int | None = None,
    show_progress: bool = True,
) -> Phase2BenchmarkReport:
    """Run the full held-out ensemble comparison across at least three seeds."""

    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) < 3:
        raise ValueError("Phase 2 reporting requires at least three seeds")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("benchmark seeds must be unique")
    _validate_phase2_epochs(vqc_epochs)

    started = time.perf_counter()
    results: list[SeedEnsembleResult] = []
    for seed in seed_values:
        if show_progress:
            print(f"Seed {seed}: training six-model quantum ensemble", flush=True)
        data = prepare_breast_cancer_data(random_state=seed)
        result = train_quantum_ensemble(
            data,
            seed=seed,
            vqc_epochs=vqc_epochs,
            learning_rate=learning_rate,
            training_sample_limit=training_sample_limit,
            show_progress=show_progress,
        )
        results.append(result)
        if show_progress:
            relation = ">=" if result.accuracy >= result.best_single_accuracy else "<"
            print(
                f"Seed {seed} complete: ensemble={result.accuracy:.4f} "
                f"{relation} best single={result.best_single_accuracy:.4f} "
                f"({result.best_single_name}); {result.elapsed_seconds:.1f}s",
                flush=True,
            )

    return Phase2BenchmarkReport(
        seed_results=tuple(results),
        seeds=seed_values,
        vqc_epochs=vqc_epochs,
        learning_rate=learning_rate,
        training_sample_limit=training_sample_limit,
        total_seconds=time.perf_counter() - started,
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join((format_row(headers), separator, *(format_row(row) for row in rows)))


def format_phase2_report(report: Phase2BenchmarkReport) -> str:
    """Format every model/weight plus honest ensemble and seed-range summaries."""

    member_rows: list[tuple[str, ...]] = []
    for result in report.seed_results:
        for member in result.members:
            member_rows.append(
                (
                    str(result.seed),
                    member.paradigm,
                    member.name,
                    f"{member.accuracy:.4f}",
                    f"{member.oob_mse:.6f}",
                    f"{member.weight:.6f}",
                    str(member.oob_size),
                )
            )

    grouped: dict[str, list[MemberEvaluation]] = {}
    for result in report.seed_results:
        for member in result.members:
            grouped.setdefault(member.name, []).append(member)
    summary_rows = []
    for name in EXPECTED_MEMBER_NAMES:
        values = grouped[name]
        accuracies = [value.accuracy for value in values]
        weights = [value.weight for value in values]
        summary_rows.append(
            (
                values[0].paradigm,
                name,
                f"{mean(accuracies):.4f}",
                f"{min(accuracies):.4f}",
                f"{max(accuracies):.4f}",
                f"{mean(weights):.6f}",
                f"{min(weights):.6f}",
                f"{max(weights):.6f}",
            )
        )

    comparison_rows = []
    for result in report.seed_results:
        comparison_rows.append(
            (
                str(result.seed),
                f"{result.accuracy:.4f}",
                result.best_single_name,
                f"{result.best_single_accuracy:.4f}",
                "yes" if result.accuracy >= result.best_single_accuracy else "no",
                f"{result.paired_comparison.statistic:.4f}",
                f"{result.paired_comparison.p_value:.6f}",
                f"{result.elapsed_seconds:.1f}",
            )
        )

    ensemble_accuracies = [result.accuracy for result in report.seed_results]
    best_accuracies = [result.best_single_accuracy for result in report.seed_results]
    overall_relation = (
        "did" if mean(ensemble_accuracies) >= mean(best_accuracies) else "did not"
    )
    pool_description = (
        "full training split"
        if report.training_sample_limit is None
        else f"stratified {report.training_sample_limit}-row training pool"
    )
    configuration = (
        "Phase 2 six-model quantum ensemble\n"
        f"Seeds: {', '.join(str(seed) for seed in report.seeds)}\n"
        f"Data: independent stratified 80/20 split per seed; {pool_description}; "
        "test rows excluded from training and OOB weighting\n"
        f"VQCs: four architecture variants, {report.vqc_epochs} full-batch Adam "
        f"epochs, learning rate {report.learning_rate}\n"
        "QSVMs: angle/4q and amplitude/2q fidelity kernels -> "
        "SVC(kernel='precomputed', probability=True)\n"
        "Weights: normalized inverse OOB probability MSE; vote: weighted soft vote\n"
        "Positive class: sklearn label 1 = benign (label 0 = malignant)"
    )
    individual = _table(
        ("seed", "type", "model", "accuracy", "OOB MSE", "weight", "OOB n"),
        member_rows,
    )
    summary = _table(
        (
            "type",
            "model",
            "acc mean",
            "acc min",
            "acc max",
            "weight mean",
            "weight min",
            "weight max",
        ),
        summary_rows,
    )
    comparisons = _table(
        (
            "seed",
            "ensemble",
            "best single",
            "single acc",
            "ensemble >=",
            "paired t",
            "p two-sided",
            "seconds",
        ),
        comparison_rows,
    )
    carried = _table(
        ("carried Phase 1 same-4 classical", "mean accuracy"),
        [
            (name, f"{accuracy:.4f}")
            for name, accuracy in CARRIED_SAME4_CLASSICAL_MEAN_ACCURACIES.items()
        ],
    )
    verdict = (
        f"Across seeds, ensemble accuracy was {mean(ensemble_accuracies):.4f} "
        f"({min(ensemble_accuracies):.4f}-{max(ensemble_accuracies):.4f}); "
        f"best-single accuracy was {mean(best_accuracies):.4f} "
        f"({min(best_accuracies):.4f}-{max(best_accuracies):.4f}). "
        f"The ensemble {overall_relation} beat/tie the best single model on "
        "mean held-out accuracy."
    )
    return (
        f"{configuration}\n\nIndividual held-out results and OOB weights\n"
        f"{individual}\n\nAcross-seed member summary\n{summary}\n\n"
        f"Ensemble versus best single model\n{comparisons}\n\n{verdict}\n\n"
        f"{carried}\n\nTotal benchmark wall time: {report.total_seconds:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_PHASE2_SEEDS),
        help="At least three unique split/model seeds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_PHASE2_EPOCHS,
        help="At least 100 Adam epochs for every VQC (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--training-limit",
        type=int,
        default=None,
        help=(
            "Optional stratified training-pool limit for timed experiments; "
            "default uses the full training split"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_phase2_benchmark(
        seeds=args.seeds,
        vqc_epochs=args.epochs,
        learning_rate=args.learning_rate,
        training_sample_limit=args.training_limit,
    )
    print()
    print(format_phase2_report(report))


if __name__ == "__main__":
    main()
