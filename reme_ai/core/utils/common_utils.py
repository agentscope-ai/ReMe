import os
import re
from pathlib import Path

from loguru import logger

ENV_LOADED = False


def camel_to_snake(content: str) -> str:
    content = content.replace("LLM", "Llm")
    snake_str = re.sub(r"(?<!^)(?=[A-Z])", "_", content).lower()
    return snake_str


def snake_to_camel(content: str) -> str:
    camel_str = "".join(x.capitalize() for x in content.split("_"))
    camel_str = camel_str.replace("Llm", "LLM")
    return camel_str


def _load_env(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue

            line_split = line.strip().split("=", 1)
            if len(line_split) >= 2:
                key = line_split[0].strip()
                value = line_split[1].strip().strip('"')
                os.environ[key] = value


def load_env(path: str | Path = None, enable_log: bool = True):
    global ENV_LOADED
    if ENV_LOADED:
        return

    if path is not None:
        path = Path(path)
        if path.exists():
            _load_env(path)
            ENV_LOADED = True

    else:
        for i in range(5):
            path = Path("../" * i + ".env")
            if path.exists():
                if enable_log:
                    logger.info(f"load env_path={path}")
                _load_env(path)
                ENV_LOADED = True
                return

        logger.warning(".env not found")


def singleton(cls):
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton
