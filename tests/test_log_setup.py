# tests/test_log_setup.py

import logging
import os

import pytest

from semanticlens.utils.log_setup import ColorFormatter, setup_colored_logging

# The name of the logger to be configured by the setup function
PACKAGE_LOGGER_NAME = "semanticlens"


@pytest.fixture(autouse=True)
def reset_logging():
    """Fixture to ensure the logging environment is clean for each test."""
    # This runs before each test
    yield
    # This runs after each test to clean up handlers
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.handlers.clear()
    logging.shutdown()


def test_setup_colored_logging_defaults(caplog):
    """
    Tests that the logger is configured with the default 'INFO' level
    and captures messages correctly.
    """
    setup_colored_logging()
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    with caplog.at_level(logging.INFO):
        logger.debug("This should not be captured.")
        logger.info("This is an info message.")

    assert logger.level == logging.INFO
    assert "This should not be captured." not in caplog.text
    assert "This is an info message." in caplog.text


def test_setup_colored_logging_sets_debug_level(caplog):
    """
    Tests that the logger level can be correctly set to 'DEBUG'.
    """
    setup_colored_logging(log_level="DEBUG")
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    with caplog.at_level(logging.DEBUG):
        logger.debug("This is a debug message.")

    assert logger.level == logging.DEBUG
    assert "This is a debug message." in caplog.text


def test_log_level_set_by_environment_variable(mocker, caplog):
    """
    Tests that the SEMANTICLENS_LOG_LEVEL environment variable overrides the default log level.
    """
    mocker.patch.dict(os.environ, {"SEMANTICLENS_LOG_LEVEL": "WARNING"})
    setup_colored_logging()  # No log_level argument provided
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    with caplog.at_level(logging.INFO):
        logger.info("This should not be captured.")
        logger.warning("This is a warning message.")

    assert logger.level == logging.WARNING
    assert "This should not be captured." not in caplog.text
    assert "This is a warning message." in caplog.text


def test_file_handler_creation(tmp_path):
    """
    Tests that a file handler is added and a log file is created
    when a file_path is provided.
    """
    log_file = tmp_path / "test.log"
    setup_colored_logging(file_path=str(log_file))
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    log_message = "This message should be in the file."
    logger.warning(log_message)

    # Ensure the file was created and contains the message
    assert log_file.is_file()
    assert log_message in log_file.read_text()


def test_color_formatter_adds_color_codes():
    """
    Tests the ColorFormatter directly to ensure it prepends and appends
    the correct ANSI color codes.
    """
    formatter = ColorFormatter("[%(levelname)s]: %(message)s", use_color=True)
    # Create a log record for testing
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname="/fake/path.py",
        lineno=10,
        msg="A test warning",
        args=(),
        exc_info=None,
    )

    formatted_message = formatter.format(record)

    assert formatted_message.startswith(ColorFormatter.COLOR_MAP["WARNING"])
    assert formatted_message.endswith(ColorFormatter.RESET_SEQ)
    assert "[WARNING]: A test warning" in formatted_message


def test_color_formatter_without_color():
    """
    Tests that the ColorFormatter does not add ANSI codes when use_color is False.
    """
    formatter = ColorFormatter("[%(levelname)s]: %(message)s", use_color=False)
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="/fake/path.py",
        lineno=10,
        msg="An info message",
        args=(),
        exc_info=None,
    )

    formatted_message = formatter.format(record)

    assert not formatted_message.startswith("\033[")
    assert not formatted_message.endswith("\033[0m")
    assert formatted_message == "[INFO]: An info message"
