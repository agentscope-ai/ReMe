import os
import sys
from datetime import datetime


def init_logger(log_dir: str = "logs", level: str = "INFO"):
    from loguru import logger

    logger.remove()
    os.makedirs(log_dir, exist_ok=True)

    current_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_filename = f"{current_ts}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger.add(
        log_filepath,
        level=level,
        rotation="00:00",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )

    logger.add(
        sink=sys.stdout,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        colorize=True,
    )
