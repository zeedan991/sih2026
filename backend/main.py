"""FastAPI service for the SIH26139 hybrid quantum-classical demo."""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.runtime import ModelRuntime, RuntimeNotReady


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"


class RuntimeContract(Protocol):
    def start_loading(self) -> None: ...

    def health_payload(self) -> dict[str, Any]: ...

    def patients_payload(self, limit: int) -> dict[str, Any]: ...

    def predict_payload(self, features: list[float]) -> dict[str, Any]: ...

    def explain_payload(
        self, features: list[float], *, allow_slow: bool
    ) -> dict[str, Any]: ...

    def baselines_payload(self) -> dict[str, Any]: ...

    def metrics_payload(self) -> dict[str, Any]: ...


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeatureRequest(StrictRequest):
    features: list[float] = Field(min_length=30, max_length=30)

    @field_validator("features")
    @classmethod
    def finite_features(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("features must contain only finite values")
        return values


class ExplainRequest(FeatureRequest):
    model: Literal["quantum"] = "quantum"
    allow_slow: bool = False


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    quantum_members: int
    classical_models: int
    selected_features: list[str]
    error: str | None = None


class PatientSummary(BaseModel):
    id: int
    name: str
    true_label: Literal["malignant", "benign"]
    features: list[float] = Field(min_length=30, max_length=30)
    selected_values: list[float] = Field(min_length=4, max_length=4)
    has_disagreement: bool


class PatientCatalogResponse(BaseModel):
    feature_names: list[str] = Field(min_length=30, max_length=30)
    selected_feature_names: list[str] = Field(min_length=4, max_length=4)
    patients: list[PatientSummary]


class PredictionLeaf(BaseModel):
    label: Literal["malignant", "benign"]
    confidence: float = Field(ge=0.5, le=1.0)
    benign_probability: float = Field(ge=0.0, le=1.0)
    model_name: str
    feature_count: int = Field(gt=0)


class QuantumMemberPrediction(BaseModel):
    name: str
    paradigm: Literal["VQC", "QSVM"]
    label: Literal["malignant", "benign"]
    confidence: float = Field(ge=0.5, le=1.0)
    benign_probability: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)


class QuantumPrediction(PredictionLeaf):
    model_count: int = Field(ge=1)
    per_model: list[QuantumMemberPrediction]


class ClassicalPrediction(BaseModel):
    primary_configuration: Literal["full_feature"]
    full_feature: PredictionLeaf
    same_4_feature: PredictionLeaf


class AgreementResult(BaseModel):
    agrees: bool
    message: str


class PredictResponse(BaseModel):
    quantum: QuantumPrediction
    classical: ClassicalPrediction
    agreement: AgreementResult


class ExplanationFeature(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    lime_value: float
    direction: Literal["toward_benign", "toward_malignant", "neutral"]


class ExplainResponse(BaseModel):
    """Attribution only: deliberately no confidence/probability field (D-23)."""

    scope: Literal["vqc_fast", "full_ensemble"]
    feature_names: list[str] = Field(min_length=4, max_length=4)
    shap_values: list[float] = Field(min_length=4, max_length=4)
    lime_values: list[float] = Field(min_length=4, max_length=4)
    directions: list[
        Literal["toward_benign", "toward_malignant", "neutral"]
    ] = Field(min_length=4, max_length=4)
    top_features: list[ExplanationFeature]
    expected_timing: str
    elapsed_seconds: float = Field(ge=0.0)


class BaselineConfiguration(BaseModel):
    feature_count: int
    runtime_seed: int | None = None
    models: dict[str, dict[str, Any]]


class BaselinesResponse(BaseModel):
    positive_class: Literal["benign"]
    runtime_split_note: str | None = None
    configurations: dict[str, BaselineConfiguration]


def create_app(*, runtime: RuntimeContract | None = None) -> FastAPI:
    """Build an app around a runtime; tests inject a non-training stub."""

    active_runtime: RuntimeContract = runtime or ModelRuntime()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        active_runtime.start_loading()
        yield

    application = FastAPI(
        title="Q-Trace | SIH26139 Hybrid Disease Detection",
        version="0.4.0",
        description=(
            "Research prototype: six-model quantum ensemble benchmarked beside "
            "classical models on the Wisconsin Breast Cancer dataset."
        ),
        lifespan=lifespan,
    )
    application.state.runtime = active_runtime
    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "QML_CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,"
            "http://localhost:8501,http://127.0.0.1:8501",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_response(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        # Removing raw inputs also avoids trying to JSON-encode NaN/Infinity
        # while explaining why a non-finite clinical value was rejected.
        safe_errors = [
            {
                key: value
                for key, value in item.items()
                if key not in {"input", "ctx"}
            }
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    def ready_call(callable_: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return callable_(*args, **kwargs)
        except RuntimeNotReady as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get(
        "/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
        tags=["system"],
    )
    def health() -> dict[str, Any]:
        return active_runtime.health_payload()

    @application.get(
        "/patients",
        response_model=PatientCatalogResponse,
        tags=["data"],
    )
    def patients(
        limit: int = Query(default=8, ge=1, le=24),
    ) -> dict[str, Any]:
        return ready_call(active_runtime.patients_payload, limit)

    @application.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
    )
    def predict(request: FeatureRequest) -> dict[str, Any]:
        return ready_call(active_runtime.predict_payload, request.features)

    @application.post(
        "/explain",
        response_model=ExplainResponse,
        tags=["explainability"],
    )
    def explain(request: ExplainRequest) -> dict[str, Any]:
        # ``model`` is deliberately constrained to quantum.  Scope is selected
        # only by allow_slow, matching architecture section 4 and D-23.
        return ready_call(
            active_runtime.explain_payload,
            request.features,
            allow_slow=request.allow_slow,
        )

    @application.get(
        "/baselines",
        response_model=BaselinesResponse,
        response_model_exclude_none=True,
        tags=["benchmarking"],
    )
    def baselines() -> dict[str, Any]:
        return ready_call(active_runtime.baselines_payload)

    @application.get("/metrics", tags=["benchmarking"])
    def metrics() -> dict[str, Any]:
        return ready_call(active_runtime.metrics_payload)

    if FRONTEND_ROOT.exists():
        application.mount(
            "/static",
            StaticFiles(directory=FRONTEND_ROOT),
            name="static",
        )

        @application.get("/", include_in_schema=False)
        def frontend() -> FileResponse:
            return FileResponse(FRONTEND_ROOT / "index.html")

    return application


app = create_app()


__all__ = [
    "ExplainResponse",
    "FeatureRequest",
    "PredictResponse",
    "app",
    "create_app",
]
