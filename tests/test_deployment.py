"""Deployment-contract checks complementary to the real Docker smoke test."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_docker_image_uses_pinned_python_stack_and_serves_fastapi() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.14-slim")
    assert "pip install -r requirements.txt" in dockerfile
    assert 'CMD ["uvicorn", "backend.main:app"' in dockerfile
    assert "COPY frontend/ ./frontend/" in dockerfile
    assert "COPY dev-dashboard/ ./dev-dashboard/" in dockerfile
    assert "ENV OMP_NUM_THREADS=1" in dockerfile


def test_compose_exposes_both_clients_and_waits_for_model_readiness() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "  api:" in compose
    assert "  dev-dashboard:" in compose
    assert 'QML_VQC_EPOCHS: "100"' in compose
    assert 'QML_TRAINING_LIMIT: "20"' in compose
    assert '"8000:8000"' in compose
    assert '"8501:8501"' in compose
    assert "condition: service_healthy" in compose
    assert "models_loaded" in compose
    assert "http://127.0.0.1:8501/_stcore/health" in compose
