import io
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from vlm_lab.api import create_app
from vlm_lab.config import Settings
from vlm_lab.domain import CaptionResult, Precision


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (12, 12), "green").save(stream, format="PNG")
    return stream.getvalue()


def test_health_models_and_caption_contract(tmp_path):
    settings = Settings(
        database_path=tmp_path / "lab.sqlite3",
        upload_dir=tmp_path / "uploads",
        data_dir=tmp_path / "data",
    )
    with TestClient(create_app(settings)) as client:
        client.app.state.services.engine.caption = Mock(
            return_value=CaptionResult(
                caption="a green square",
                model_id="blip-base",
                precision=Precision.FULL,
                latency_ms=4.2,
                process_memory_mb=90,
            )
        )
        assert client.get("/api/v1/health").json()["status"] == "ok"
        assert len(client.get("/api/v1/models").json()) == 2
        response = client.post(
            "/api/v1/caption",
            files={"image": ("square.png", image_bytes(), "image/png")},
            data={"model_id": "blip-base", "precision": "full"},
        )
        assert response.status_code == 200
        assert response.json()["caption"] == "a green square"


def test_caption_error_contract(tmp_path):
    settings = Settings(
        database_path=tmp_path / "lab.sqlite3",
        upload_dir=tmp_path / "uploads",
        data_dir=tmp_path / "data",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/caption",
            files={"image": ("bad.txt", b"bad", "text/plain")},
            data={"model_id": "blip-base", "precision": "full"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_IMAGE"
        assert "error" in response.json()


def test_create_experiment_validates_manifest_root(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "test.csv"
    manifest.write_text("image_path,references\n", "utf-8")
    settings = Settings(
        database_path=tmp_path / "lab.sqlite3",
        upload_dir=tmp_path / "uploads",
        data_dir=data_dir,
    )
    with TestClient(create_app(settings)) as client:
        client.app.state.services.runner.submit = Mock()
        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "test run",
                "dataset_manifest": str(manifest),
                "model_ids": ["blip-base"],
                "precisions": ["full"],
                "sample_limit": 1,
            },
        )
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
