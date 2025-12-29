import os
from pathlib import Path

from loguru import logger

ENV_LOADED = False


def _load_env(path: Path) -> None:
    """
    Load environment variables from a file.
    
    Args:
        path: Path to the .env file
    """
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Split on first '=' only
                line_split = line.split("=", 1)
                if len(line_split) >= 2:
                    key = line_split[0].strip()
                    value = line_split[1].strip()
                    
                    # Remove quotes (both single and double)
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    os.environ[key] = value
    except PermissionError as e:
        logger.warning(f"Permission denied when accessing env file {path}: {e}. Running in sandbox mode?")
    except Exception as e:
        logger.error(f"Failed to load env file {path}: {e}")
        raise


def load_env(path: str | Path | None = None, enable_log: bool = True) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        path: Path to .env file. If None, searches up to 5 parent directories.
        enable_log: Whether to log the loaded path.
    """
    global ENV_LOADED
    if ENV_LOADED:
        return

    if path is not None:
        path = Path(path)
        if path.exists():
            _load_env(path)
            ENV_LOADED = True
        else:
            logger.warning(f".env file not found at specified path: {path}")
    else:
        # Search in current and parent directories
        current_dir = Path.cwd()
        for _ in range(5):
            env_path = current_dir / ".env"
            if env_path.exists():
                if enable_log:
                    logger.info(f"load env_path={env_path}")
                _load_env(env_path)
                ENV_LOADED = True
                return
            current_dir = current_dir.parent

        logger.warning(".env not found in current or parent directories")


def reset_env_loaded() -> None:
    """Reset ENV_LOADED flag. Useful for testing."""
    global ENV_LOADED
    ENV_LOADED = False
