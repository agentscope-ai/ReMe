"""Standard logging module configuration with loguru-like features."""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Store created logger instances
_loggers: dict[str, logging.Logger] = {}


class CustomFormatter(logging.Formatter):
    """Custom formatter with colorized output support."""

    # ANSI color codes
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, colorize: bool = False):
        super().__init__(fmt)
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        # Add custom attribute: simplified filename and line number
        record.file_line = f"{record.filename}:{record.lineno}"

        if self.colorize:
            color = self.COLORS.get(record.levelno, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


def get_loggerv2(
    name: str = "reme",
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_file_prefix: str = "reme",
    rotation: str = "midnight",
    retention_days: int = 7,
    force_update: bool = False,
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name for distinguishing different loggers.
        log_dir: Directory path for log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_console: Whether to output logs to console.
        log_to_file: Whether to output logs to file.
        log_file_prefix: Prefix for log file names (e.g., 'reme' -> 'reme_2024-01-01.log').
        rotation: Log rotation time, defaults to midnight.
        retention_days: Number of days to retain log files.
        force_update: Whether to force update the logger configuration even if it already exists.

    Returns:
        Configured Logger instance.
    """
    # Return existing logger if already created and not force updating
    if name in _loggers and not force_update:
        return _loggers[name]

    # Create new logger without using root logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # Do not propagate to root logger

    # Clear existing handlers
    logger.handlers.clear()

    # Log format
    log_format = "%(asctime)s | %(levelname)s | %(file_line)s | %(funcName)s | %(message)s"

    # Configure file logging
    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            current_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = f"{log_file_prefix}_{current_ts}.log"
            log_filepath = os.path.join(log_dir, log_filename)

            file_handler = TimedRotatingFileHandler(
                log_filepath,
                when=rotation,
                interval=1,
                backupCount=retention_days,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            file_handler.setFormatter(CustomFormatter(log_format, colorize=False))
            file_handler.suffix = "%Y-%m-%d"
            logger.addHandler(file_handler)

        except Exception as e:
            logger.error(f"Error configuring file logging: {e}")

    # Configure console logging
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(CustomFormatter(log_format, colorize=True))
        logger.addHandler(console_handler)

    # Cache logger
    _loggers[name] = logger
    return logger


def get_logger():
    """Get a configured logger instance using loguru."""
    from loguru import logger

    return logger
