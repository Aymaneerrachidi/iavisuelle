from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock

from .datasets import load_manifest
from .domain import ExperimentCreate, JobStatus
from .images import load_local_image
from .inference import CaptionEngine
from .repository import Repository

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, repository: Repository, engine: CaptionEngine, workers: int = 1) -> None:
        self.repository = repository
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm-worker")
        self._cancellations: dict[str, Event] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()

    def submit(self, experiment_id: str, config: ExperimentCreate) -> None:
        cancellation = Event()
        with self._lock:
            self._cancellations[experiment_id] = cancellation
            self._futures[experiment_id] = self.executor.submit(
                self._run, experiment_id, config, cancellation
            )

    def cancel(self, experiment_id: str) -> bool:
        with self._lock:
            cancellation = self._cancellations.get(experiment_id)
        if cancellation is None:
            return False
        cancellation.set()
        return True

    def _run(self, experiment_id: str, config: ExperimentCreate, cancellation: Event) -> None:
        try:
            items = load_manifest(config.dataset_manifest, config.sample_limit)
            actual_total = len(items) * len(config.model_ids) * len(config.precisions)
            with self.repository.connection() as db:
                db.execute(
                    "UPDATE experiments SET progress_total = ? WHERE id = ?",
                    (actual_total, experiment_id),
                )
            self.repository.set_status(experiment_id, JobStatus.RUNNING)
            for model_id in config.model_ids:
                for precision in config.precisions:
                    for item in items:
                        if cancellation.is_set():
                            self.repository.set_status(experiment_id, JobStatus.CANCELLED)
                            return
                        image = load_local_image(item.image_path)
                        result = self.engine.caption(image, model_id, precision, config.options)
                        self.repository.add_prediction(
                            experiment_id,
                            {
                                **result.model_dump(mode="json"),
                                "image_id": item.image_id,
                                "image_path": item.image_path,
                                "references": item.references,
                            },
                        )
                        self.repository.increment_progress(experiment_id)
            self.repository.set_status(experiment_id, JobStatus.COMPLETED)
        except Exception as exc:  # worker boundary: sanitize before API exposure
            logger.exception("Experiment %s failed", experiment_id)
            self.repository.set_status(
                experiment_id,
                JobStatus.FAILED,
                error=f"Experiment failed: {type(exc).__name__}. Check the server logs.",
            )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
