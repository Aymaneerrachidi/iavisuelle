from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import Experiment, ExperimentCreate, JobStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    image_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    references_json TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    precision TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    peak_vram_mb REAL,
                    process_memory_mb REAL,
                    UNIQUE(experiment_id, image_id, model_id, precision)
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_experiment
                ON predictions(experiment_id);
                """
            )

    def recover_interrupted(self) -> None:
        """Make stale jobs explicit after an unclean server restart."""
        with self.connection() as db:
            db.execute(
                """UPDATE experiments
                SET status = ?, error = ?, finished_at = ?
                WHERE status IN (?, ?)""",
                (
                    JobStatus.FAILED,
                    "Experiment interrupted by a server restart. Create a new run to retry.",
                    utc_now(),
                    JobStatus.QUEUED,
                    JobStatus.RUNNING,
                ),
            )

    def create_experiment(self, experiment_id: str, config: ExperimentCreate) -> Experiment:
        created = utc_now()
        total = config.sample_limit * len(config.model_ids) * len(config.precisions)
        with self.connection() as db:
            db.execute(
                """INSERT INTO experiments
                (id, name, status, config_json, progress_total, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    config.name,
                    JobStatus.QUEUED,
                    config.model_dump_json(),
                    total,
                    created,
                ),
            )
        return self.get_experiment(experiment_id)

    @staticmethod
    def _to_experiment(row: sqlite3.Row) -> Experiment:
        return Experiment(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            config=ExperimentCreate.model_validate_json(row["config_json"]),
            progress_current=row["progress_current"],
            progress_total=row["progress_total"],
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def get_experiment(self, experiment_id: str) -> Experiment:
        with self.connection() as db:
            row = db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return self._to_experiment(row)

    def list_experiments(self, limit: int = 50) -> list[Experiment]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_experiment(row) for row in rows]

    def set_status(self, experiment_id: str, status: JobStatus, error: str | None = None) -> None:
        updates: dict[str, Any] = {"status": status.value, "error": error}
        if status == JobStatus.RUNNING:
            updates["started_at"] = utc_now()
        if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            updates["finished_at"] = utc_now()
        clauses = ", ".join(f"{key} = ?" for key in updates)
        with self.connection() as db:
            db.execute(
                f"UPDATE experiments SET {clauses} WHERE id = ?",  # noqa: S608
                (*updates.values(), experiment_id),
            )

    def increment_progress(self, experiment_id: str) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE experiments SET progress_current = progress_current + 1 WHERE id = ?",
                (experiment_id,),
            )

    def add_prediction(self, experiment_id: str, prediction: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute(
                """INSERT OR REPLACE INTO predictions
                (experiment_id, image_id, image_path, references_json, model_id, precision,
                 caption, latency_ms, peak_vram_mb, process_memory_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    prediction["image_id"],
                    prediction["image_path"],
                    json.dumps(prediction["references"], ensure_ascii=False),
                    prediction["model_id"],
                    prediction["precision"],
                    prediction["caption"],
                    prediction["latency_ms"],
                    prediction.get("peak_vram_mb"),
                    prediction.get("process_memory_mb"),
                ),
            )

    def predictions(self, experiment_id: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM predictions WHERE experiment_id = ? ORDER BY id", (experiment_id,)
            ).fetchall()
        return [
            {
                **dict(row),
                "references": json.loads(row["references_json"]),
            }
            for row in rows
        ]
