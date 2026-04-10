"""Logging configuration module for application-wide tracing."""

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

    Args:
        log_dir: Directory path for log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_console: Whether to print logs to console/screen.
        log_to_file: Whether to write logs to file.
        force_init: Force re-initialization even if already initialized.

    Returns:
        The configured logger instance.
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
                rotation="00:00",
                retention="7 days",
                compression="zip",
                encoding="utf-8",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}",
            )
        except Exception as e:
            logger.error(f"Error configuring file logging: {e}")

    _initialized = True
    return logger