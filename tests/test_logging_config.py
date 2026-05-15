"""Tests for logging configuration."""
from __future__ import annotations

import logging

import pytest

from app.logging_config import configure_logging, get_logger


class TestConfigureLogging:
    def test_sets_root_level_info(self) -> None:
        configure_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_sets_root_level_debug(self) -> None:
        configure_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_level_falls_back(self) -> None:
        configure_logging("NOTEXIST")
        assert logging.getLogger().level == logging.INFO

    def test_adds_handler(self) -> None:
        configure_logging("INFO")
        assert len(logging.getLogger().handlers) >= 1


class TestGetLogger:
    def test_returns_logger(self) -> None:
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_name_preserved(self) -> None:
        logger = get_logger("my.custom.logger")
        assert logger.name == "my.custom.logger"

    @pytest.mark.parametrize("name", ["app.main", "insurance_predictor", "scripts"])
    def test_various_names(self, name: str) -> None:
        logger = get_logger(name)
        assert logger.name == name
