"""Logging configuration module for application-wide tracing.

Provides a centralized logging facility using Loguru with support for
both console and file-based output, automatic rotation, and retention.
"""

import os
import sys
from datetime import datetime

from loguru import logger

_initialized = False


def get_logger(
        log_dir: str = "logs",
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        force_init: bool = False,
):
    """Get a configured logger instance.

    Automatically initializes on first call. Subsequent calls return
    the same logger without re-initializing unless force_init=True.

    This function configures the global Loguru logger with:
    - Colorized console output (optional)
    - File output with daily rotation (optional)
    - 7-day retention with ZIP compression
    - Consistent timestamp and location formatting

    Args:
        log_dir: Directory path for log files. Created if it doesn't exist.
            Defaults to "logs" in the current working directory.
        level: Logging level threshold. One of: DEBUG, INFO, WARNING,
            ERROR, CRITICAL. Defaults to "INFO".
        log_to_console: Whether to print logs to stdout. Defaults to True.
        log_to_file: Whether to write logs to file. Defaults to True.
        force_init: Force re-initialization even if already initialized.
            Useful for changing log configuration mid-application.
            Defaults to False.

    Returns:
        The configured Loguru logger instance.
    """
    global _initialized

    if _initialized and not force_init:
        return logger

    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Configure colorized console logging if enabled
    if log_to_console:
        logger.add(
            sink=sys.stdout,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}",
            colorize=True,
        )

    # Configure file-based logging if enabled
    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)

            current_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filepath = os.path.join(log_dir, f"{current_ts}.log")

            logger.add(
                log_filepath,
                level=level,
                rotation="00:00",  # Rotate at midnight
                retention="7 days",  # Keep logs for 7 days
                compression="zip",  # Compress rotated logs
                encoding="utf-8",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}",
            )
        except Exception as e:
            logger.error(f"Error configuring file logging: {e}")

    _initialized = True
    return logger
