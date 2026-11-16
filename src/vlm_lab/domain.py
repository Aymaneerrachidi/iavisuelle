from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Precision(StrEnum):
    FULL = "full"
    INT8 = "int8"
    INT4 = "int4"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelInfo(BaseModel):
    id: str
    label: str
    hub_id: str
    architecture: str
    parameters_millions: int
    supported_precisions: list[Precision]
    license: str


class GenerationOptions(BaseModel):
    max_new_tokens: int = Field(default=40, ge=5, le=128)
    num_beams: int = Field(default=3, ge=1, le=8)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)


class CaptionResult(BaseModel):
    caption: str
    model_id: str
    precision: Precision
    latency_ms: float
    peak_vram_mb: float | None = None
    process_memory_mb: float | None = None


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    dataset_manifest: str = Field(min_length=1, max_length=500)
    model_ids: list[str] = Field(min_length=1, max_length=2)
    precisions: list[Precision] = Field(min_length=1, max_length=3)
    sample_limit: int = Field(default=100, ge=1, le=2000)
    options: GenerationOptions = GenerationOptions()


class Experiment(BaseModel):
    id: str
    name: str
    status: JobStatus
    config: ExperimentCreate
    progress_current: int = 0
    progress_total: int = 0
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class MetricSummary(BaseModel):
    model_id: str
    precision: Precision
    samples: int
    bleu4: float | None = None
    rouge_l: float | None = None
    mean_latency_ms: float
    p95_latency_ms: float
    peak_vram_mb: float | None = None
    mean_caption_words: float


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class DatasetItem:
    image_path: str
    references: list[str]
    image_id: str
