"""Create the retained three-seed evidence record for the diabetes module."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from backend.classical.baselines import evaluate_classification, train_evaluate_baselines
from backend.data.diabetes import prepare_diabetes_data
from backend.quantum.train import train_quantum_ensemble


SEEDS = (42, 123, 2026)
OUTPUT = Path("artifacts/evaluation/early_diabetes_three_seed.json")


def range_summary(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def main() -> None:
    runs = []
    member_values: dict[str, list[float]] = {}
    member_weights: dict[str, list[float]] = {}
    classical_values: dict[str, list[float]] = {}
    for seed in SEEDS:
        print(f"Seed {seed}: early-diabetes six-model ensemble", flush=True)
        data = prepare_diabetes_data(random_state=seed)
        classical = train_evaluate_baselines(data, random_state=seed)
        quantum = train_quantum_ensemble(
            data,
            seed=seed,
            vqc_epochs=100,
            training_sample_limit=20,
            show_progress=True,
        )
        metrics = evaluate_classification(
            data.y_test,
            quantum.predictions,
            malignant_scores=1.0 - quantum.positive_probabilities,
            positive_class_name="positive screening signal",
        )
        members = []
        for member in quantum.members:
            member_values.setdefault(member.name, []).append(member.accuracy)
            member_weights.setdefault(member.name, []).append(member.weight)
            members.append(
                {
                    "name": member.name,
                    "paradigm": member.paradigm,
                    "accuracy": member.accuracy,
                    "oob_mse": member.oob_mse,
                    "weight": member.weight,
                    "oob_size": member.oob_size,
                }
            )
        classical_rows = {}
        for configuration, models in classical.items():
            classical_rows[configuration] = {}
            for name, result in models.items():
                key = f"{configuration}/{name}"
                classical_values.setdefault(key, []).append(result.metrics.accuracy)
                classical_rows[configuration][name] = asdict(result.metrics)
        runs.append(
            {
                "seed": seed,
                "selected_feature_names": data.selected_feature_names.tolist(),
                "ensemble_metrics": asdict(metrics),
                "best_single_name": quantum.best_single_name,
                "best_single_accuracy": quantum.best_single_accuracy,
                "members": members,
                "classical": classical_rows,
            }
        )
        print(
            f"Seed {seed} complete: ensemble={quantum.accuracy:.4f}; "
            f"sensitivity={metrics.condition_sensitivity:.4f}; "
            f"specificity={metrics.specificity:.4f}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "dataset": "UCI Early Stage Diabetes Risk Prediction",
        "source": "https://doi.org/10.24432/C5VG8H",
        "license": "CC BY 4.0",
        "seeds": list(SEEDS),
        "configuration": {
            "split": "stratified 80/20",
            "vqc_epochs": 100,
            "quantum_training_limit": 20,
            "quantum_device": "lightning.qubit",
            "preprocessing": "train-only median -> standard -> SelectKBest(k=4) -> inward quantum range",
        },
        "runs": runs,
        "summaries": {
            "ensemble_accuracy": range_summary(
                [run["ensemble_metrics"]["accuracy"] for run in runs]
            ),
            "ensemble_condition_sensitivity": range_summary(
                [run["ensemble_metrics"]["condition_sensitivity"] for run in runs]
            ),
            "ensemble_specificity": range_summary(
                [run["ensemble_metrics"]["specificity"] for run in runs]
            ),
            "members": {
                name: {
                    "accuracy": range_summary(values),
                    "weight": range_summary(member_weights[name]),
                }
                for name, values in member_values.items()
            },
            "classical_accuracy": {
                name: range_summary(values)
                for name, values in classical_values.items()
            },
        },
        "qualification": (
            "A small single-center symptom questionnaire can yield high apparent "
            "accuracy. This record demonstrates pipeline reuse; it is not external, "
            "prospective, Indian-population, fairness, or clinical validation."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2), flush=True)
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
