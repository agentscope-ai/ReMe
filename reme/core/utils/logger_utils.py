"""Logging configuration module for application-wide tracing."""

import os
import sys
from datetime import datetime


def init_logger(log_dir: str = "logs", level: str = "INFO", log_to_console: bool = True) -> None:
    """Initialize the logger with both file and console handlers.

    Args:
        log_dir: Directory path for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to print logs to console/screen
    """
    from loguru import logger

    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate filename based on the current timestamp
    # Use dashes instead of colons for Windows compatibility
    current_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_ts}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Configure file-based logging with rotation and compression
    logger.add(
        log_filepath,
        level=level,
        rotation="00:00",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}",
    )

    # Configure colorized standard output logging if enabled
    if log_to_console:
        logger.add(
            sink=sys.stdout,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}",
            colorize=True,
        )
