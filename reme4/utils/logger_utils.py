"""Logger utilities supporting both loguru and standard logging backends."""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

_logger = None

_LOGURU_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}"
_STDLIB_FORMAT = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
_STDLIB_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _enable_loguru() -> bool:
    return os.getenv("REME_DISABLE_LOGURU", "").lower() != "true"


def _init_loguru(log_dir: str, level: str, log_to_console: bool, log_to_file: bool):
    from loguru import logger

    logger.remove()

    if log_to_console:
        logger.add(
            sink=sys.stdout,
            level=level,
            format=_LOGURU_FORMAT,
            colorize=True,
        )

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
                format=_LOGURU_FORMAT,
            )
        except Exception as e:
            logger.error(f"Error configuring file logging: {e}")

    return logger


def _init_stdlib(log_dir: str, level: str, log_to_console: bool, log_to_file: bool):
    logger = logging.getLogger("reme")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(_STDLIB_FORMAT, datefmt=_STDLIB_DATEFMT)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            current_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filepath = os.path.join(log_dir, f"{current_ts}.log")

            file_handler = TimedRotatingFileHandler(
                log_filepath,
                when="midnight",
                backupCount=7,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Error configuring file logging: {e}")

    return logger


def get_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    force_init: bool = False,
):
    """Return the global logger, initializing sinks on first call (or when force_init)."""
    global _logger

    if _logger is not None and not force_init:
        return _logger

    if _enable_loguru():
        _logger = _init_loguru(log_dir, level, log_to_console, log_to_file)
    else:
        _logger = _init_stdlib(log_dir, level, log_to_console, log_to_file)
    return _logger
