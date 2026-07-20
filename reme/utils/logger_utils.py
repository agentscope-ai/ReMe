"""Logger utilities supporting both loguru and standard logging backends."""

import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

_logger = None
_logger_lock = threading.RLock()

_LOGURU_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {function} | {message}"
_STDLIB_FORMAT = "%(levelname)s %(source_path)s:%(lineno)d | %(asctime)s | %(message)s"
_STDLIB_DATEFMT = "%Y-%m-%d %H:%M:%S"
_QWENPAW_LOGGER_NAME = "qwenpaw"


class _ForwardToLoggerHandler(logging.Handler):
    """Forward records to a host logger without borrowing its handlers."""

    def __init__(self, target_name: str) -> None:
        super().__init__()
        self.target_name = target_name

    def emit(self, record: logging.LogRecord) -> None:
        target = logging.getLogger(self.target_name)
        if target.isEnabledFor(record.levelno):
            target.handle(record)


class _QwenPawStdlibFormatter(logging.Formatter):
    """Format stdlib records consistently with QwenPaw host logs."""

    def format(self, record: logging.LogRecord) -> str:
        source_path = record.pathname
        cwd = os.getcwd()
        try:
            if os.path.commonpath([source_path, cwd]) == cwd:
                source_path = os.path.relpath(source_path, cwd)
        except ValueError:
            # Paths on different Windows drives cannot be compared.
            pass

        # QwenPaw prefixes console records with a cwd-relative source path.
        record.source_path = source_path
        return super().format(record)


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
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    qwenpaw_logger = logging.getLogger(_QWENPAW_LOGGER_NAME)
    if qwenpaw_logger.handlers:
        # QwenPaw owns the screen and file handlers. Forwarding keeps ReMe's
        # logger object stable for modules that cache it at import time, while
        # allowing future QwenPaw handlers (for example qwenpaw.log) to take
        # effect without another ReMe reconfiguration.
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_ForwardToLoggerHandler(_QWENPAW_LOGGER_NAME))
        return logger

    logger.setLevel(level)

    formatter = _QwenPawStdlibFormatter(_STDLIB_FORMAT, datefmt=_STDLIB_DATEFMT)

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

    # ReMe can be embedded multiple times in one process.  Hosts may construct
    # those applications concurrently, while both logging backends reconfigure
    # a process-global logger via a remove-then-add sequence.  Keep the whole
    # check/reconfigure/publish transaction atomic so concurrent force_init
    # calls cannot leave duplicate sinks or handlers behind.
    with _logger_lock:
        if _logger is not None and not force_init:
            return _logger

        if _enable_loguru():
            _logger = _init_loguru(log_dir, level, log_to_console, log_to_file)
        else:
            _logger = _init_stdlib(log_dir, level, log_to_console, log_to_file)
        return _logger
