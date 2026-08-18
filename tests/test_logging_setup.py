"""Tests for the central `dryrun._logging` setup."""

from __future__ import annotations

import logging

from dryrun._logging import dryrun_logger, setup_logging


class TestSetupLogging:
    def test_returns_the_shared_dryrun_logger(self):
        logger = setup_logging(force=True)
        assert logger is dryrun_logger
        assert logger.name == "dryrun"

    def test_idempotent_by_default(self):
        setup_logging(force=True)
        n_handlers = len(dryrun_logger.handlers)
        setup_logging()  # no force -> should not add more handlers.
        assert len(dryrun_logger.handlers) == n_handlers

    def test_force_reconfigures(self):
        setup_logging(level="DEBUG", force=True)
        assert dryrun_logger.level == logging.DEBUG
        setup_logging(level="WARNING", force=True)
        assert dryrun_logger.level == logging.WARNING

    def test_formatter_includes_timestamp_and_source_location(self):
        setup_logging(force=True)
        handler = dryrun_logger.handlers[0]
        fmt = handler.formatter._fmt
        assert "%(asctime)s" in fmt
        assert "%(filename)s" in fmt
        assert "%(lineno)d" in fmt
        assert "%(levelname)s" in fmt

    def test_does_not_propagate_to_root(self):
        setup_logging(force=True)
        assert dryrun_logger.propagate is False

    def test_string_level_is_case_insensitive(self):
        setup_logging(level="debug", force=True)
        assert dryrun_logger.level == logging.DEBUG

    def test_log_file_also_writes_to_disk(self, tmp_path):
        log_file = tmp_path / "run.log"
        setup_logging(level="INFO", log_file=log_file, force=True)
        dryrun_logger.info("hello from a test")
        for handler in dryrun_logger.handlers:
            handler.flush()
        assert log_file.exists()
        assert "hello from a test" in log_file.read_text()

    def test_env_var_default_level(self, monkeypatch):
        monkeypatch.setenv("DRYRUN_LOG_LEVEL", "ERROR")
        setup_logging(force=True)
        assert dryrun_logger.level == logging.ERROR
