"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from environment variables.

    All values have sensible defaults so the app works without a .env file.
    """

    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    uvicorn_host: str = field(default_factory=lambda: os.getenv("UVICORN_HOST", "0.0.0.0"))
    uvicorn_port: int = field(default_factory=lambda: int(os.getenv("UVICORN_PORT", "8000")))
    uvicorn_workers: int = field(
        default_factory=lambda: int(os.getenv("UVICORN_WORKERS", "1"))
    )
    model_path: pathlib.Path | None = field(
        default_factory=lambda: (
            pathlib.Path(p) if (p := os.getenv("MODEL_PATH")) else None
        )
    )
    data_path: pathlib.Path | None = field(
        default_factory=lambda: (
            pathlib.Path(p) if (p := os.getenv("DATA_PATH")) else None
        )
    )
    rate_limit_requests: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    )
    rate_limit_window: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    )

    @property
    def is_production(self) -> bool:
        """True when running in a production environment."""
        return self.app_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
