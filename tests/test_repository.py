from vlm_lab.domain import ExperimentCreate, JobStatus, Precision
from vlm_lab.repository import Repository


def config():
    return ExperimentCreate(
        name="contract run",
        dataset_manifest="data/processed/test.csv",
        model_ids=["blip-base"],
        precisions=[Precision.FULL],
        sample_limit=2,
    )


def test_repository_lifecycle(tmp_path):
    repository = Repository(tmp_path / "lab.sqlite3")
    repository.initialize()
    experiment = repository.create_experiment("run-1", config())
    assert experiment.status == JobStatus.QUEUED
    assert experiment.progress_total == 2

    repository.set_status("run-1", JobStatus.RUNNING)
    repository.increment_progress("run-1")
    repository.add_prediction(
        "run-1",
        {
            "image_id": "one",
            "image_path": "/tmp/one.jpg",
            "references": ["one image"],
            "model_id": "blip-base",
            "precision": "full",
            "caption": "an image",
            "latency_ms": 12.5,
            "peak_vram_mb": None,
            "process_memory_mb": 100,
        },
    )
    repository.set_status("run-1", JobStatus.COMPLETED)

    saved = repository.get_experiment("run-1")
    assert saved.status == JobStatus.COMPLETED
    assert saved.progress_current == 1
    assert repository.predictions("run-1")[0]["caption"] == "an image"


def test_recover_interrupted_jobs(tmp_path):
    repository = Repository(tmp_path / "lab.sqlite3")
    repository.initialize()
    repository.create_experiment("run-1", config())
    repository.recover_interrupted()
    assert repository.get_experiment("run-1").status == JobStatus.FAILED
