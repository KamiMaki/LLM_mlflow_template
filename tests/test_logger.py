"""app.logger 單元測試。"""

from __future__ import annotations

from app.logger import get_logger, setup_logging
from app.utils.config import init_config


class TestSetupLogging:
    def test_setup_without_config(self):
        setup_logging()

    def test_setup_with_config(self):
        cfg = init_config()
        setup_logging(cfg)


class TestGetLogger:
    def test_returns_logger(self):
        setup_logging()
        logger = get_logger(__name__)
        assert logger is not None

    def test_logger_can_log(self):
        setup_logging()
        logger = get_logger(__name__)
        logger.info("test message")
