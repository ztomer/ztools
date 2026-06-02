"""Tests for lib.logging_config."""
import logging
import pytest
from pathlib import Path


class TestGetLogger:
    def test_basic_logger(self):
        from lib.logging_config import get_logger
        # Reset cache
        log = logging.getLogger("test_basic")
        log.handlers = []
        result = get_logger("test_basic")
        assert result is not None
        assert result.name == "test_basic"
        assert result.level == logging.INFO

    def test_logger_with_debug_level(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_debug")
        log.handlers = []
        result = get_logger("test_debug", level="DEBUG")
        assert result.level == logging.DEBUG

    def test_logger_with_warning_level(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_warning")
        log.handlers = []
        result = get_logger("test_warning", level="WARNING")
        assert result.level == logging.WARNING

    def test_logger_with_error_level(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_error")
        log.handlers = []
        result = get_logger("test_error", level="ERROR")
        assert result.level == logging.ERROR

    def test_logger_with_critical_level(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_critical")
        log.handlers = []
        result = get_logger("test_critical", level="CRITICAL")
        assert result.level == logging.CRITICAL

    def test_logger_invalid_level(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_invalid_level")
        log.handlers = []
        result = get_logger("test_invalid_level", level="BOGUS")
        # Falls back to INFO
        assert result.level == logging.INFO

    def test_logger_cached(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_cached")
        log.handlers = []
        first = get_logger("test_cached")
        # Second call returns the same logger (no double-config)
        second = get_logger("test_cached")
        assert first is second

    def test_logger_with_file(self, tmp_path):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_file")
        log.handlers = []
        log_file = tmp_path / "test.log"
        get_logger("test_file", level="DEBUG", log_file=log_file)
        assert log_file.parent.exists()

    def test_logger_without_console(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_no_console")
        log.handlers = []
        result = get_logger("test_no_console", console_output=False)
        # Should have at most a file handler (none in this case)
        # No console handler added
        assert result.propagate is False

    def test_logger_propagate_false(self):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_propagate")
        log.handlers = []
        result = get_logger("test_propagate")
        assert result.propagate is False

    def test_logger_writes_to_file(self, tmp_path):
        from lib.logging_config import get_logger
        log = logging.getLogger("test_write")
        log.handlers = []
        log_file = tmp_path / "writes.log"
        logger = get_logger("test_write", level="INFO", log_file=log_file, console_output=False)
        logger.info("test message")
        # File should have content
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message" in content


class TestModuleLevel:
    def test_lib_logger(self):
        from lib.logging_config import lib_logger
        assert lib_logger is not None
        assert lib_logger.name == "lib"

    def test_osaurus_logger(self):
        from lib.logging_config import osaurus_logger
        assert osaurus_logger is not None
        assert osaurus_logger.name == "lib.osaurus"

    def test_mlx_logger(self):
        from lib.logging_config import mlx_logger
        assert mlx_logger is not None
        assert mlx_logger.name == "lib.mlx"

    def test_validators_logger(self):
        from lib.logging_config import validators_logger
        assert validators_logger is not None
        assert validators_logger.name == "lib.validators"

    def test_content_logger(self):
        from lib.logging_config import content_logger
        assert content_logger is not None
        assert content_logger.name == "lib.content"


class TestConstants:
    def test_log_levels_keys(self):
        from lib.logging_config import LOG_LEVELS
        assert "DEBUG" in LOG_LEVELS
        assert "INFO" in LOG_LEVELS
        assert "WARNING" in LOG_LEVELS
        assert "ERROR" in LOG_LEVELS
        assert "CRITICAL" in LOG_LEVELS

    def test_log_format(self):
        from lib.logging_config import LOG_FORMAT
        assert "%(asctime)s" in LOG_FORMAT
        assert "%(levelname)s" in LOG_FORMAT

    def test_date_format(self):
        from lib.logging_config import DATE_FORMAT
        assert "%Y" in DATE_FORMAT
