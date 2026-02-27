"""Environment variable loader utility for managing .env files."""

import os
from pathlib import Path

from loguru import logger

# Global flag to ensure environment is loaded only once
_ENV_LOADED = False


def _parse_env_file(path: Path) -> None:
    """Parse and inject key-value pairs from a .env file into os.environ."""
    try:
        with path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" in line:
                    key, value = line.split("=", 1)
                    # Strip whitespace and common quotes
                    os.environ[key.strip()] = value.strip().strip("'\"")
    except PermissionError as err:
        logger.warning(f"Permission denied for {path}: {err}")
    except Exception as err:
        logger.error(f"Failed to load {path}: {err}")
        raise


def load_env(path: str | Path | None = None, enable_log: bool = True) -> None:
    """Search and load the .env file into the system environment."""
    global _ENV_LOADED  # pylint: disable=global-statement
    if _ENV_LOADED:
        return

    if path:
        path = Path(path)
        if path.exists():
            _parse_env_file(path)
            _ENV_LOADED = True
        else:
            logger.warning(f".env not found at: {path}")
        return

    # Search current directory and up to 5 levels of parents
    for directory in [Path.cwd(), *Path.cwd().parents[:5]]:
        env_path = directory / ".env"
        if env_path.exists():
            if enable_log:
                logger.info(f"Loading environment from: {env_path}")
            _parse_env_file(env_path)
            _ENV_LOADED = True
            return


def reset_env_flag() -> None:
    """Reset the internal load state flag."""
    global _ENV_LOADED  # pylint: disable=global-statement
    _ENV_LOADED = False
