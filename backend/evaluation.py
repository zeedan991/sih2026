"""Leakage-safe generalization and computational-efficiency evaluation.

Every fold rebuilds imputation, scaling, ANOVA feature selection, and quantum
range scaling from its training rows.  Malignant (sklearn label 0) is always
the positive clinical condition for sensitivity, F1, and ROC-AUC.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from backend.classical.baselines import (
    MODEL_NAMES,
    ClassificationMetrics,
    evaluate_classification,
    train_evaluate_baselines,
)
from backend.data.pipeline import prepare_breast_cancer_data, prepare_breast_cancer_partition
from backend.quantum.train import MINIMUM_PHASE2_EPOCHS, train_quantum_ensemble


DEFAULT_CV_FOLDS = 5
DEFAULT_CV_SEED = 42
DEFAULT_EVIDENCE_SEEDS = (42, 123, 2026)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    fold_or_seed: int
    paradigm: str
    feature_configuration: str
    model_name: str
    metrics: ClassificationMetrics
    fit_seconds: float | None
    predict_seconds: float | None
    timing_scope: str


@dataclass(frozen=True, slots=True)
class CrossValidationFold:
    fold: int
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_labels: tuple[int, ...]
    preprocessing_fit_rows: int
    selected_feature_names: tuple[str, ...]
    rows: tuple[EvaluationRow, ...]


@dataclass(frozen=True, slots=True)
class CrossValidationReport:
    n_splits: int
    random_state: int
    includes_quantum: bool
    quantum_training_limit: int | None
    vqc_epochs: int | None
    folds: tuple[CrossValidationFold, ...]
    summaries: dict[str, dict[str, Any]]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RepeatedClassicalReport:
    seeds: tuple[int, ...]
    rows: tuple[EvaluationRow, ...]
    summaries: dict[str, dict[str, Any]]
    elapsed_seconds: float


def _summaries(rows: Sequence[EvaluationRow]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        key = f"{row.paradigm}/{row.feature_configuration}/{row.model_name}"
        grouped.setdefault(key, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, values in grouped.items():
        metric_names = (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "malignant_sensitivity",
            "specificity",
            "roc_auc",
        )
        metrics: dict[str, Any] = {}
        for metric_name in metric_names:
            observed = [
                float(value)
                for row in values
                if (value := getattr(row.metrics, metric_name)) is not None
            ]
            metrics[metric_name] = {
                "mean": mean(observed),
                "range": [min(observed), max(observed)],
            }
        fit_timings = [row.fit_seconds for row in values if row.fit_seconds is not None]
        predict_timings = [
            row.predict_seconds for row in values if row.predict_seconds is not None
        ]
        result[key] = {
            "runs": len(values),
            "metrics": metrics,
            "mean_fit_seconds": mean(fit_timings) if fit_timings else None,
            "mean_predict_seconds": mean(predict_timings) if predict_timings else None,
            "timing_scope": sorted({row.timing_scope for row in values}),
            "total_false_negatives": sum(
                row.metrics.confusion_matrix["false_negative"] for row in values
            ),
        }
    return result


def _classical_rows(
    prepared: Any,
    *,
    identifier: int,
    random_state: int,
    model_names: tuple[str, ...],
) -> list[EvaluationRow]:
    results = train_evaluate_baselines(
        prepared,
        random_state=random_state,
        model_names=model_names,
    )
    return [
        EvaluationRow(
            fold_or_seed=identifier,
            paradigm="classical",
            feature_configuration=configuration,
            model_name=model_name,
            metrics=result.metrics,
            fit_seconds=result.fit_seconds,
            predict_seconds=result.predict_seconds,
            timing_scope="individual_model_fit_and_heldout_prediction",
        )
        for configuration, models in results.items()
        for model_name, result in models.items()
    ]


def run_repeated_classical_benchmark(
    *,
    seeds: Sequence[int] = DEFAULT_EVIDENCE_SEEDS,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> RepeatedClassicalReport:
    """Retain complete three-seed metrics for both classical feature views."""

    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) < 3 or len(set(seed_values)) != len(seed_values):
        raise ValueError("classical evidence requires at least three unique seeds")
    started = time.perf_counter()
    rows: list[EvaluationRow] = []
    for seed in seed_values:
        prepared = prepare_breast_cancer_data(random_state=seed)
        rows.extend(
            _classical_rows(
                prepared,
                identifier=seed,
                random_state=seed,
                model_names=model_names,
            )
        )
    return RepeatedClassicalReport(
        seeds=seed_values,
        rows=tuple(rows),
        summaries=_summaries(rows),
        elapsed_seconds=time.perf_counter() - started,
    )


def run_cross_validation(
    *,
    n_splits: int = DEFAULT_CV_FOLDS,
    random_state: int = DEFAULT_CV_SEED,
    include_quantum: bool = True,
    quantum_training_limit: int = 20,
    vqc_epochs: int = MINIMUM_PHASE2_EPOCHS,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> CrossValidationReport:
    """Run stratified fold validation with all preprocessing fitted per fold."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if include_quantum and vqc_epochs < MINIMUM_PHASE2_EPOCHS:
        raise ValueError(
            "quantum cross-validation requires at least "
            f"{MINIMUM_PHASE2_EPOCHS} VQC epochs"
        )
    dataset = load_breast_cancer()
    indices = np.arange(dataset.target.size, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    started = time.perf_counter()
    folds: list[CrossValidationFold] = []
    all_rows: list[EvaluationRow] = []
    for fold_index, (training, validation) in enumerate(
        splitter.split(dataset.data, dataset.target), start=1
    ):
        training_indices = indices[training]
        validation_indices = indices[validation]
        prepared = prepare_breast_cancer_partition(training_indices, validation_indices)
        rows = _classical_rows(
            prepared,
            identifier=fold_index,
            random_state=random_state + fold_index,
            model_names=model_names,
        )
        if include_quantum:
            quantum = train_quantum_ensemble(
                prepared,
                seed=random_state + fold_index,
                vqc_epochs=vqc_epochs,
                training_sample_limit=quantum_training_limit,
                show_progress=False,
                device_name="lightning.qubit",
            )
            quantum_members = [*quantum.members]
            for member in quantum_members:
                rows.append(
                    EvaluationRow(
                        fold_or_seed=fold_index,
                        paradigm="quantum",
                        feature_configuration="same_4_quantum_range",
                        model_name=member.name,
                        metrics=evaluate_classification(
                            prepared.y_test,
                            member.predictions,
                            malignant_scores=1.0 - member.positive_probabilities,
                        ),
                        fit_seconds=member.elapsed_seconds,
                        predict_seconds=None,
                        timing_scope=(
                            "individual_quantum_member_training_oob_and_internal_"
                            "heldout_evaluation"
                        ),
                    )
                )
            rows.append(
                EvaluationRow(
                    fold_or_seed=fold_index,
                    paradigm="quantum",
                    feature_configuration="same_4_quantum_range",
                    model_name="six_model_oob_ensemble",
                    metrics=evaluate_classification(
                        prepared.y_test,
                        quantum.predictions,
                        malignant_scores=1.0 - quantum.positive_probabilities,
                    ),
                    fit_seconds=quantum.elapsed_seconds,
                    predict_seconds=None,
                    timing_scope=(
                        "complete_six_model_training_oob_weighting_and_internal_"
                        "heldout_evaluation"
                    ),
                )
            )
        all_rows.extend(rows)
        folds.append(
            CrossValidationFold(
                fold=fold_index,
                training_indices=tuple(int(value) for value in training_indices),
                validation_indices=tuple(int(value) for value in validation_indices),
                validation_labels=tuple(int(value) for value in prepared.y_test),
                preprocessing_fit_rows=int(training_indices.size),
                selected_feature_names=tuple(
                    str(value) for value in prepared.selected_feature_names
                ),
                rows=tuple(rows),
            )
        )
    return CrossValidationReport(
        n_splits=n_splits,
        random_state=random_state,
        includes_quantum=include_quantum,
        quantum_training_limit=quantum_training_limit if include_quantum else None,
        vqc_epochs=vqc_epochs if include_quantum else None,
        folds=tuple(folds),
        summaries=_summaries(all_rows),
        elapsed_seconds=time.perf_counter() - started,
    )


def write_report(report: CrossValidationReport | RepeatedClassicalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate retained SIH26139 evaluation evidence"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--quantum-training-limit", type=int, default=20)
    arguments = parser.parse_args()
    classical = run_repeated_classical_benchmark()
    cross_validation = run_cross_validation(
        include_quantum=not arguments.skip_quantum,
        quantum_training_limit=arguments.quantum_training_limit,
    )
    write_report(classical, arguments.output_dir / "classical_three_seed_metrics.json")
    write_report(cross_validation, arguments.output_dir / "five_fold_cross_validation.json")
    print(json.dumps({
        "classical_seconds": classical.elapsed_seconds,
        "cross_validation_seconds": cross_validation.elapsed_seconds,
        "cross_validation_summaries": cross_validation.summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
