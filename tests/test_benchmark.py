"""Configuration checks for reproducible benchmark runs."""

from backend.benchmark import DEFAULT_BENCHMARK_EPOCHS, DEFAULT_SEEDS


def test_default_benchmark_uses_converged_training_budget() -> None:
    assert DEFAULT_BENCHMARK_EPOCHS >= 100
    assert DEFAULT_SEEDS == (42, 123, 2026)
