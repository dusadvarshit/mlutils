import logging

import pytest

from mlutils.utils.logger import CustomLogger


@pytest.fixture
def logger_instance():
    return CustomLogger(name="test_logger", level="DEBUG")


def test_logger_initialization(logger_instance):
    logger = logger_instance.get_logger()
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG


def test_logger_change_level(logger_instance):
    logger_instance.change_level("WARNING")
    logger = logger_instance.get_logger()
    assert logger.level == logging.WARNING


def test_invalid_level_defaults_to_info():
    logger_instance = CustomLogger(name="invalid_level_logger", level="INVALID")
    logger = logger_instance.get_logger()
    assert logger.level == logging.INFO


def test_logger_outputs_to_stream(caplog):
    logger_instance = CustomLogger(name="stream_logger", level="INFO")
    logger = logger_instance.get_logger()

    with caplog.at_level(logging.INFO):
        logger.info("Hello from logger")

    assert "Hello from logger" in caplog.text
