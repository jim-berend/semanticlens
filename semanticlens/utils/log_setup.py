"""Logging configuration for the SemanticLens package.

This module provides a centralized, optional configuration for colored
console logging. It is designed to be non-intrusive, allowing the user
of the library to enable and configure logging without interfering with
their own application's logging setup.

The primary function, `setup_colored_logging`, attaches a formatted
handler to the top-level 'semanticlens' logger. By default, the library
has a `NullHandler` to prevent "No handler found" warnings if the user
does not configure logging.

Functions
---------
setup_colored_logging
    Configures a colored, stream-based logger for the package.


Example
-------
>>> from semanticlens.logging import setup_colored_logging
>>> setup_colored_logging("DEBUG", "path-to-logs/debug.log)
"""

from __future__ import annotations

import logging
import os

PACKAGE = "semanticlens"

assert __package__.startswith(PACKAGE), f"Package name mismatch: {__package__} does not start with {PACKAGE}"


class ColorFormatter(logging.Formatter):
    """A custom formatter to add colors to log messages."""

    COLOR_MAP = {
        "DEBUG": "\033[90m",
        "INFO": "\033[92m",
        "WARNING": "\033[38;5;208m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[91m",  # "\033[95m",
    }
    RESET_SEQ = "\033[0m"

    def __init__(self, fmt, use_color=True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record):
        """Format the specified record as text."""
        record.short_filename = os.path.basename(record.pathname)
        message = super().format(record)
        if self.use_color:
            color = self.COLOR_MAP.get(record.levelname, "")
            return f"{color}{message}{self.RESET_SEQ}"
        return message


def setup_colored_logging(log_level: str = "INFO", file_path: str | None = None):
    """Configures a colored logger for the 'semanticlens' package."""
    logger = logging.getLogger(PACKAGE)
    effective_level_str = os.environ.get("SEMANTICLENS_LOG_LEVEL", log_level).upper()
    effective_level = getattr(logging, effective_level_str, logging.INFO)
    logger.setLevel(effective_level)

    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(effective_level)
    use_color = hasattr(handler.stream, "isatty") and handler.stream.isatty()

    formatter = ColorFormatter(
        "[%(asctime)s|%(name)s|%(levelname)s]: %(message)s",
        use_color=use_color,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(
            ColorFormatter(
                "[%(asctime)s|%(name)s|%(levelname)s]: %(message)s",
                use_color=False,
            )
        )
        logger.addHandler(file_handler)


logging.getLogger(PACKAGE).addHandler(logging.NullHandler())
