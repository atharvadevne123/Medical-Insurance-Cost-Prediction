"""Tests for application configuration."""
from __future__ import annotations

import pytest

from app.config import Settings, get_settings


class TestSettings:
    def test_defaults_are_set(self) -> None:
        s = Settings()
        assert s.app_env == "development"
        assert s.log_level == "INFO"
        assert s.uvicorn_host == "0.0.0.0"
        assert s.uvicorn_port == 8000

    def test_is_production_false_by_default(self) -> None:
        s = Settings()
        assert s.is_production is False

    def test_get_settings_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_frozen(self) -> None:
        s = Settings()
        with pytest.raises((AttributeError, TypeError)):
            s.log_level = "DEBUG"  # type: ignore[misc]

    @pytest.mark.parametrize("env,expected", [
        ("production", True),
        ("development", False),
        ("test", False),
    ])
    def test_is_production_from_app_env(self, env: str, expected: bool) -> None:
        class _S(Settings):
            app_env: str = env
        s = _S.__new__(_S)
        object.__setattr__(s, "app_env", env)
        assert s.is_production is expected
