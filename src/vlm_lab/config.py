from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLM_LAB_", env_file=".env", extra="ignore")

    env: str = "development"
    database_path: Path = Path("artifacts/vlm_lab.sqlite3")
    upload_dir: Path = Path("data/uploads")
    data_dir: Path = Path("data/processed")
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    max_image_pixels: int = Field(default=25_000_000, ge=1_000_000)
    workers: int = Field(default=1, ge=1, le=4)
    model_cache_size: int = Field(default=1, ge=0, le=3)
    cors_origins: list[str] = []
    log_level: str = "INFO"

    def prepare_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
