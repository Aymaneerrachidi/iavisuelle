from __future__ import annotations

import csv
import io
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .domain import (
    CaptionResult,
    ErrorResponse,
    Experiment,
    ExperimentCreate,
    GenerationOptions,
    JobStatus,
    MetricSummary,
    Precision,
)
from .experiments import ExperimentRunner
from .images import InvalidImage, decode_image
from .inference import CaptionEngine
from .metrics import summarize
from .model_registry import MODELS, get_model
from .repository import Repository

logger = logging.getLogger(__name__)


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = Repository(settings.database_path)
        self.engine = CaptionEngine(settings.model_cache_size)
        self.runner = ExperimentRunner(self.repository, self.engine, settings.workers)


def services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


def problem(
    status: int, error: str, code: str, details: dict[str, Any] | None = None
) -> NoReturn:
    raise HTTPException(
        status_code=status, detail={"error": error, "code": code, "details": details}
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings.prepare_directories()
        app.state.services = Services(active_settings)
        app.state.services.repository.initialize()
        app.state.services.repository.recover_interrupted()
        yield
        app.state.services.runner.shutdown()

    app = FastAPI(
        title="VLM Lab API",
        version="1.0.0",
        description="Benchmark two lightweight captioning VLMs across full, 8-bit and 4-bit modes.",
        lifespan=lifespan,
    )
    if active_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=active_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            body = exc.detail
        else:
            body = {"error": str(exc.detail), "code": "HTTP_ERROR", "details": None}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = {
            "fields": [
                {"path": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
                for error in exc.errors()
            ]
        }
        return JSONResponse(
            status_code=422,
            content={
                "error": "Request validation failed.",
                "code": "VALIDATION_ERROR",
                "details": details,
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"},
        )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/api/v1/models")
    def models() -> list[dict[str, Any]]:
        return [model.model_dump(mode="json") for model in MODELS.values()]

    @app.post(
        "/api/v1/caption",
        response_model=CaptionResult,
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def caption(
        image: Annotated[UploadFile, File()],
        model_id: Annotated[str, Form()] = "blip-base",
        precision: Annotated[Precision, Form()] = Precision.FULL,
        max_new_tokens: Annotated[int, Form(ge=5, le=128)] = 40,
        num_beams: Annotated[int, Form(ge=1, le=8)] = 3,
        service: Services = Depends(services),
    ) -> CaptionResult:
        try:
            info = get_model(model_id)
        except ValueError:
            problem(400, "Unknown model.", "MODEL_NOT_FOUND")
        if precision not in info.supported_precisions:
            problem(400, "Precision is not supported by this model.", "PRECISION_UNSUPPORTED")
        limit = service.settings.max_upload_mb * 1024 * 1024
        payload = await image.read(limit + 1)
        if len(payload) > limit:
            problem(413, "Image is larger than the configured limit.", "FILE_TOO_LARGE")
        try:
            decoded = decode_image(payload, service.settings.max_image_pixels)
        except InvalidImage as exc:
            problem(400, str(exc), "INVALID_IMAGE")
        try:
            return service.engine.caption(
                decoded,
                model_id,
                precision,
                GenerationOptions(max_new_tokens=max_new_tokens, num_beams=num_beams),
            )
        except RuntimeError as exc:
            problem(503, str(exc), "INFERENCE_UNAVAILABLE")

    @app.post("/api/v1/experiments", response_model=Experiment, status_code=202)
    def create_experiment(
        payload: ExperimentCreate, service: Services = Depends(services)
    ) -> Experiment:
        unknown = [model_id for model_id in payload.model_ids if model_id not in MODELS]
        if unknown:
            problem(400, "One or more models are unknown.", "MODEL_NOT_FOUND", {"ids": unknown})
        manifest = Path(payload.dataset_manifest).expanduser().resolve()
        if not manifest.is_file():
            problem(400, "Dataset manifest does not exist on the server.", "MANIFEST_NOT_FOUND")
        data_root = active_settings.data_dir.resolve()
        if not manifest.is_relative_to(data_root):
            problem(
                400,
                "Dataset manifest must be inside the configured data directory.",
                "MANIFEST_OUTSIDE_DATA_ROOT",
            )
        experiment_id = str(uuid.uuid4())
        experiment = service.repository.create_experiment(experiment_id, payload)
        service.runner.submit(experiment_id, payload)
        return experiment

    @app.get("/api/v1/experiments", response_model=list[Experiment])
    def list_experiments(service: Services = Depends(services)) -> list[Experiment]:
        return service.repository.list_experiments()

    @app.get("/api/v1/experiments/{experiment_id}", response_model=Experiment)
    def get_experiment_route(
        experiment_id: str, service: Services = Depends(services)
    ) -> Experiment:
        try:
            return service.repository.get_experiment(experiment_id)
        except KeyError:
            problem(404, "Experiment not found.", "EXPERIMENT_NOT_FOUND")

    @app.post("/api/v1/experiments/{experiment_id}/cancel", response_model=Experiment)
    def cancel_experiment(experiment_id: str, service: Services = Depends(services)) -> Experiment:
        try:
            experiment = service.repository.get_experiment(experiment_id)
        except KeyError:
            problem(404, "Experiment not found.", "EXPERIMENT_NOT_FOUND")
        if experiment.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            problem(409, "Only queued or running experiments can be cancelled.", "INVALID_STATE")
        if not service.runner.cancel(experiment_id):
            problem(409, "The experiment worker is no longer active.", "WORKER_INACTIVE")
        return service.repository.get_experiment(experiment_id)

    @app.get("/api/v1/experiments/{experiment_id}/metrics", response_model=list[MetricSummary])
    def experiment_metrics(
        experiment_id: str, service: Services = Depends(services)
    ) -> list[MetricSummary]:
        try:
            service.repository.get_experiment(experiment_id)
        except KeyError:
            problem(404, "Experiment not found.", "EXPERIMENT_NOT_FOUND")
        return summarize(service.repository.predictions(experiment_id))

    @app.get("/api/v1/experiments/{experiment_id}/predictions")
    def experiment_predictions(
        experiment_id: str, limit: int = 100, service: Services = Depends(services)
    ) -> dict[str, Any]:
        if limit < 1 or limit > 500:
            problem(400, "Limit must be between 1 and 500.", "VALIDATION_ERROR")
        try:
            service.repository.get_experiment(experiment_id)
        except KeyError:
            problem(404, "Experiment not found.", "EXPERIMENT_NOT_FOUND")
        rows = service.repository.predictions(experiment_id)
        return {"items": rows[:limit], "total": len(rows)}

    @app.get("/api/v1/experiments/{experiment_id}/export.csv")
    def export_experiment(experiment_id: str, service: Services = Depends(services)) -> Response:
        try:
            service.repository.get_experiment(experiment_id)
        except KeyError:
            problem(404, "Experiment not found.", "EXPERIMENT_NOT_FOUND")
        rows = service.repository.predictions(experiment_id)
        stream = io.StringIO()
        fields = [
            "image_id",
            "image_path",
            "model_id",
            "precision",
            "caption",
            "latency_ms",
            "peak_vram_mb",
            "process_memory_mb",
            "references_json",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{experiment_id}.csv"'},
        )

    web_dir = Path(__file__).parent / "web"
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


app = create_app()
