"""FastAPI service for the SIH26139 hybrid quantum-classical demo."""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.runtime import ModelRuntime, RuntimeNotReady
from backend.data.diseases import DEFAULT_DISEASE_ID
from backend.request_limits import RequestSizeLimit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"


class RuntimeContract(Protocol):
    def start_loading(self) -> None: ...

    def health_payload(self) -> dict[str, Any]: ...

    def diseases_payload(self) -> dict[str, Any]: ...

    def patients_payload(
        self, limit: int, *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]: ...

    def predict_payload(
        self, features: list[float], *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]: ...

    def ingest_payload(
        self, record: dict[str, float], *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]: ...

    def explain_payload(
        self,
        features: list[float],
        *,
        model: str,
        allow_slow: bool,
        disease_id: str = DEFAULT_DISEASE_ID,
    ) -> dict[str, Any]: ...

    def baselines_payload(self, *, disease_id: str = DEFAULT_DISEASE_ID) -> dict[str, Any]: ...

    def metrics_payload(self, *, disease_id: str = DEFAULT_DISEASE_ID) -> dict[str, Any]: ...

    def report_payload(
        self, features: list[float], *, disease_id: str = DEFAULT_DISEASE_ID
    ) -> dict[str, Any]: ...


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeatureRequest(StrictRequest):
    disease_id: str = Field(default=DEFAULT_DISEASE_ID, min_length=1, max_length=64)
    features: list[float] = Field(min_length=4, max_length=64)

    @field_validator("features")
    @classmethod
    def finite_features(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("features must contain only finite values")
        return values

    @model_validator(mode="after")
    def disease_specific_shape(self) -> "FeatureRequest":
        expected = {"breast_cancer": 30, "early_diabetes": 16}.get(self.disease_id)
        if expected is not None and len(self.features) != expected:
            raise ValueError(
                f"{self.disease_id} requires exactly {expected} feature values"
            )
        return self


class ExplainRequest(FeatureRequest):
    model: Literal[
        "quantum", "classical_full_feature", "classical_same_4_feature"
    ] = "quantum"
    allow_slow: bool = False


class NamedRecordRequest(StrictRequest):
    disease_id: str = Field(default=DEFAULT_DISEASE_ID, min_length=1, max_length=64)
    record: dict[str, float] = Field(min_length=4, max_length=64)

    @field_validator("record")
    @classmethod
    def finite_record(cls, values: dict[str, float]) -> dict[str, float]:
        if not all(
            name.strip() and math.isfinite(value) for name, value in values.items()
        ):
            raise ValueError("record names must be non-empty and values must be finite")
        return values

    @model_validator(mode="after")
    def disease_specific_shape(self) -> "NamedRecordRequest":
        expected = {"breast_cancer": 30, "early_diabetes": 16}.get(self.disease_id)
        if expected is not None and len(self.record) != expected:
            raise ValueError(
                f"{self.disease_id} requires exactly {expected} named features"
            )
        return self


class RangeWarning(BaseModel):
    feature_name: str
    value: float
    observed_min: float
    observed_max: float


class IngestResponse(BaseModel):
    disease_id: str = DEFAULT_DISEASE_ID
    feature_names: list[str] = Field(min_length=4, max_length=64)
    features: list[float] = Field(min_length=4, max_length=64)
    warnings: list[RangeWarning]


class RuntimeConfiguration(BaseModel):
    configuration_id: str | None = None
    manifest: str | None = None
    loading_mode: str | None = None
    quantum_device: str | None = None
    seed: int
    vqc_epochs: int
    quantum_training_limit: int
    classical_training_rows: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    quantum_members: int
    classical_models: int
    selected_features: list[str]
    disease_modules: list[dict[str, Any]] = Field(default_factory=list)
    runtime_configuration: RuntimeConfiguration
    error: str | None = None


class PatientSummary(BaseModel):
    id: int
    name: str
    true_label: str
    features: list[float] = Field(min_length=4, max_length=64)
    selected_values: list[float] = Field(min_length=4, max_length=4)
    has_disagreement: bool


class PatientCatalogResponse(BaseModel):
    disease_id: str = DEFAULT_DISEASE_ID
    disease_title: str | None = None
    dataset_name: str | None = None
    class_labels: list[str] = Field(default_factory=list)
    feature_names: list[str] = Field(min_length=4, max_length=64)
    selected_feature_names: list[str] = Field(min_length=4, max_length=4)
    patients: list[PatientSummary]


class PredictionLeaf(BaseModel):
    label: str
    confidence: float = Field(ge=0.5, le=1.0)
    class_one_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    benign_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    model_name: str
    feature_count: int = Field(gt=0)


class QuantumMemberPrediction(BaseModel):
    name: str
    paradigm: Literal["VQC", "QSVM"]
    label: str
    confidence: float = Field(ge=0.5, le=1.0)
    class_one_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    benign_probability: float | None = Field(default=None, ge=0.0, le=1.0)
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
    disease_id: str = DEFAULT_DISEASE_ID
    disease_title: str | None = None
    dataset_name: str | None = None
    class_labels: list[str] = Field(default_factory=list)
    quantum: QuantumPrediction
    classical: ClassicalPrediction
    agreement: AgreementResult


class ExplanationFeature(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    lime_value: float
    direction: str


class ExplainResponse(BaseModel):
    """Attribution only: deliberately no confidence/probability field (D-23)."""

    disease_id: str = DEFAULT_DISEASE_ID
    scope: Literal[
        "vqc_fast",
        "full_ensemble",
        "classical_full_feature",
        "classical_same_4_feature",
    ]
    feature_names: list[str] = Field(min_length=4, max_length=30)
    shap_values: list[float] = Field(min_length=4, max_length=30)
    lime_values: list[float] = Field(min_length=4, max_length=30)
    directions: list[str] = Field(min_length=4, max_length=30)
    top_features: list[ExplanationFeature]
    expected_timing: str
    elapsed_seconds: float = Field(ge=0.0)


class BaselineConfiguration(BaseModel):
    feature_count: int
    runtime_seed: int | None = None
    models: dict[str, dict[str, Any]]


class BaselinesResponse(BaseModel):
    disease_id: str = DEFAULT_DISEASE_ID
    positive_class: str
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
        version="0.6.0",
        description=(
            "Multi-disease research prototype: a six-model quantum ensemble "
            "benchmarked beside classical models on two biomedical datasets."
        ),
        lifespan=lifespan,
    )
    application.state.runtime = active_runtime
    # One expensive operation per process. Do not fill the HTTP thread pool
    # with requests waiting behind a 70s+ explanation; health stays available.
    inference_slot = BoundedSemaphore(1)
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
    application.add_middleware(RequestSizeLimit)

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

    def inference_call(callable_: Any, *args: Any, **kwargs: Any) -> Any:
        if not inference_slot.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Another prediction or explanation is running. Please retry shortly.",
                headers={"Retry-After": "2"},
            )
        try:
            return ready_call(callable_, *args, **kwargs)
        finally:
            inference_slot.release()

    @application.get(
        "/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
        tags=["system"],
    )
    def health() -> dict[str, Any]:
        return active_runtime.health_payload()

    @application.get("/diseases", tags=["data"])
    def diseases() -> dict[str, Any]:
        return ready_call(active_runtime.diseases_payload)

    @application.get(
        "/patients",
        response_model=PatientCatalogResponse,
        tags=["data"],
    )
    def patients(
        limit: int = Query(default=8, ge=1, le=24),
        disease_id: str = Query(default=DEFAULT_DISEASE_ID),
    ) -> dict[str, Any]:
        if disease_id == DEFAULT_DISEASE_ID:
            return ready_call(active_runtime.patients_payload, limit)
        try:
            return ready_call(
                active_runtime.patients_payload, limit, disease_id=disease_id
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post(
        "/ingest",
        response_model=IngestResponse,
        tags=["data"],
    )
    def ingest(request: NamedRecordRequest) -> dict[str, Any]:
        try:
            if request.disease_id == DEFAULT_DISEASE_ID:
                return ready_call(active_runtime.ingest_payload, request.record)
            return ready_call(
                active_runtime.ingest_payload,
                request.record,
                disease_id=request.disease_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post(
        "/predict",
        response_model=PredictResponse,
        response_model_exclude_none=True,
        response_model_exclude_unset=True,
        tags=["inference"],
    )
    def predict(request: FeatureRequest) -> dict[str, Any]:
        try:
            if request.disease_id == DEFAULT_DISEASE_ID:
                return inference_call(active_runtime.predict_payload, request.features)
            return inference_call(
                active_runtime.predict_payload,
                request.features,
                disease_id=request.disease_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post(
        "/explain",
        response_model=ExplainResponse,
        response_model_exclude_none=True,
        tags=["explainability"],
    )
    def explain(request: ExplainRequest) -> dict[str, Any]:
        try:
            kwargs = {"model": request.model, "allow_slow": request.allow_slow}
            if request.disease_id != DEFAULT_DISEASE_ID:
                kwargs["disease_id"] = request.disease_id
            return inference_call(active_runtime.explain_payload, request.features, **kwargs)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get(
        "/baselines",
        response_model=BaselinesResponse,
        response_model_exclude_none=True,
        tags=["benchmarking"],
    )
    def baselines(
        disease_id: str = Query(default=DEFAULT_DISEASE_ID),
    ) -> dict[str, Any]:
        if disease_id == DEFAULT_DISEASE_ID:
            return ready_call(active_runtime.baselines_payload)
        try:
            return ready_call(active_runtime.baselines_payload, disease_id=disease_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/metrics", tags=["benchmarking"])
    def metrics(
        disease_id: str = Query(default=DEFAULT_DISEASE_ID),
    ) -> dict[str, Any]:
        if disease_id == DEFAULT_DISEASE_ID:
            return ready_call(active_runtime.metrics_payload)
        try:
            return ready_call(active_runtime.metrics_payload, disease_id=disease_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/report", tags=["reporting"])
    def report(request: FeatureRequest) -> dict[str, Any]:
        try:
            if request.disease_id == DEFAULT_DISEASE_ID:
                return inference_call(active_runtime.report_payload, request.features)
            return inference_call(
                active_runtime.report_payload,
                request.features,
                disease_id=request.disease_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

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
    "IngestResponse",
    "PredictResponse",
    "app",
    "create_app",
]
