"""Roadmap Phase 1 multi-seed VQC/classical benchmark.

This module deliberately benchmarks only the single architecture-specified
VQC and the eight classical model/configuration combinations.  QSVMs, the
quantum ensemble, API, explainability, and frontend belong to later phases.

Run from the repository root with::

    python -m backend.benchmark

Clinical metrics use class 0 (malignant) as positive while preserving sklearn's
original WBCD encoding. Every seed gets its own stratified split and train-fitted
preprocessing pipeline; no test rows participate in fitting or training.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from statistics import mean
from typing import Final, Sequence

from backend.classical.baselines import (
    ClassificationMetrics,
    evaluate_classification,
    train_evaluate_baselines,
)
from backend.data.pipeline import prepare_breast_cancer_data
from backend.quantum.vqc_circuits import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_LAYERS,
    DEFAULT_N_QUBITS,
    predict_vqc,
    train_vqc,
)


DEFAULT_SEEDS: Final[tuple[int, ...]] = (42, 123, 2026)
DEFAULT_BENCHMARK_EPOCHS: Final[int] = 100


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    seed: int
    paradigm: str
    model_name: str
    feature_configuration: str
    metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class SeedTiming:
    seed: int
    vqc_seconds: float
    classical_seconds: float
    total_seconds: float
    initial_vqc_cost: float
    final_vqc_cost: float


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    rows: tuple[BenchmarkRow, ...]
    timings: tuple[SeedTiming, ...]
    total_seconds: float
    seeds: tuple[int, ...]
    vqc_epochs: int
    learning_rate: float


def run_benchmark(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    vqc_epochs: int = DEFAULT_BENCHMARK_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> BenchmarkReport:
    """Run a leakage-safe benchmark across at least three random seeds."""

    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) < 3:
        raise ValueError("benchmark reporting requires at least three seeds")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("benchmark seeds must be unique")
    if vqc_epochs < 1:
        raise ValueError("vqc_epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    benchmark_started = time.perf_counter()
    rows: list[BenchmarkRow] = []
    timings: list[SeedTiming] = []

    for seed in seed_values:
        seed_started = time.perf_counter()
        data = prepare_breast_cancer_data(random_state=seed)

        vqc_started = time.perf_counter()
        vqc = train_vqc(
            data.X_train_quantum,
            data.y_train_pm1,
            seed=seed,
            n_qubits=DEFAULT_N_QUBITS,
            n_layers=DEFAULT_N_LAYERS,
            n_epochs=vqc_epochs,
            learning_rate=learning_rate,
        )
        vqc_predictions = predict_vqc(
            vqc.circuit,
            vqc.raw_weights,
            data.X_test_quantum,
            n_qubits=DEFAULT_N_QUBITS,
        )
        vqc_seconds = time.perf_counter() - vqc_started
        rows.append(
            BenchmarkRow(
                seed=seed,
                paradigm="quantum",
                model_name="vqc",
                feature_configuration="same_4_quantum_range",
                metrics=evaluate_classification(data.y_test, vqc_predictions),
            )
        )

        classical_started = time.perf_counter()
        classical_results = train_evaluate_baselines(data, random_state=seed)
        classical_seconds = time.perf_counter() - classical_started
        for configuration, model_results in classical_results.items():
            for model_name, result in model_results.items():
                rows.append(
                    BenchmarkRow(
                        seed=seed,
                        paradigm="classical",
                        model_name=model_name,
                        feature_configuration=configuration,
                        metrics=result.metrics,
                    )
                )

        total_seed_seconds = time.perf_counter() - seed_started
        timing = SeedTiming(
            seed=seed,
            vqc_seconds=vqc_seconds,
            classical_seconds=classical_seconds,
            total_seconds=total_seed_seconds,
            initial_vqc_cost=vqc.cost_history[0],
            final_vqc_cost=vqc.cost_history[-1],
        )
        timings.append(timing)
        print(
            f"Seed {seed} complete in {timing.total_seconds:.1f}s "
            f"(VQC {timing.vqc_seconds:.1f}s, classical "
            f"{timing.classical_seconds:.1f}s; VQC cost "
            f"{timing.initial_vqc_cost:.6f} -> {timing.final_vqc_cost:.6f})",
            flush=True,
        )

    return BenchmarkReport(
        rows=tuple(rows),
        timings=tuple(timings),
        total_seconds=time.perf_counter() - benchmark_started,
        seeds=seed_values,
        vqc_epochs=vqc_epochs,
        learning_rate=learning_rate,
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


def format_report(report: BenchmarkReport) -> str:
    """Format per-seed results and an across-seed mean/min/max summary."""

    per_seed_rows = [
        (
            str(row.seed),
            row.paradigm,
            row.model_name,
            row.feature_configuration,
            f"{row.metrics.accuracy:.4f}",
            f"{row.metrics.precision:.4f}",
            f"{row.metrics.recall:.4f}",
            f"{row.metrics.f1:.4f}",
        )
        for row in report.rows
    ]

    grouped: dict[tuple[str, str, str], list[ClassificationMetrics]] = {}
    for row in report.rows:
        key = (row.paradigm, row.model_name, row.feature_configuration)
        grouped.setdefault(key, []).append(row.metrics)

    summary_rows: list[tuple[str, ...]] = []
    for (paradigm, model_name, configuration), metrics_values in grouped.items():
        accuracies = [metrics.accuracy for metrics in metrics_values]
        summary_rows.append(
            (
                paradigm,
                model_name,
                configuration,
                f"{mean(accuracies):.4f}",
                f"{min(accuracies):.4f}",
                f"{max(accuracies):.4f}",
                f"{mean(metrics.precision for metrics in metrics_values):.4f}",
                f"{mean(metrics.recall for metrics in metrics_values):.4f}",
                f"{mean(metrics.f1 for metrics in metrics_values):.4f}",
            )
        )

    config = (
        "Phase 1 benchmark configuration\n"
        f"Seeds: {', '.join(str(seed) for seed in report.seeds)}\n"
        f"Split: stratified 80/20, preprocessing fit per seed on train rows only\n"
        f"VQC: {DEFAULT_N_QUBITS} qubits, {DEFAULT_N_LAYERS} layers, "
        f"data re-uploading, sigmoid weight remapping, {report.vqc_epochs} "
        f"full-batch Adam epochs, learning rate {report.learning_rate}\n"
        "Metrics: positive label 1 = benign (0 = malignant)"
    )
    per_seed = _table(
        (
            "seed",
            "paradigm",
            "model",
            "features",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ),
        per_seed_rows,
    )
    summary = _table(
        (
            "paradigm",
            "model",
            "features",
            "acc mean",
            "acc min",
            "acc max",
            "precision mean",
            "recall mean",
            "f1 mean",
        ),
        summary_rows,
    )
    timing_rows = [
        (
            str(timing.seed),
            f"{timing.vqc_seconds:.1f}",
            f"{timing.classical_seconds:.1f}",
            f"{timing.total_seconds:.1f}",
            f"{timing.initial_vqc_cost:.6f}",
            f"{timing.final_vqc_cost:.6f}",
        )
        for timing in report.timings
    ]
    timings = _table(
        ("seed", "VQC sec", "classical sec", "total sec", "cost 0", "cost final"),
        timing_rows,
    )
    return (
        f"{config}\n\nPer-seed held-out metrics\n{per_seed}\n\n"
        f"Across-seed summary\n{summary}\n\nTiming and VQC training check\n"
        f"{timings}\nTotal benchmark wall time: {report.total_seconds:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="At least three unique train/test and model seeds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_BENCHMARK_EPOCHS,
        help="Full-batch Adam epochs for each VQC (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        seeds=args.seeds,
        vqc_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print()
    print(format_report(report))


if __name__ == "__main__":
    main()
