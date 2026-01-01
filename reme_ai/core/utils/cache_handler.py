"""Local file-based cache utility for DataFrames, lists, dicts, and strings."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class CacheHandler:
    """Handles persistent data caching with expiration and type support."""

    _EXTENSIONS = {
        pd.DataFrame: ".csv",
        dict: ".json",
        list: ".json",
        str: ".txt",
    }

    _TYPE_NAMES = {
        "DataFrame": pd.DataFrame,
        "dict": dict,
        "list": list,
        "str": str,
    }

    def __init__(self, cache_dir: str | Path = "cache"):
        """Initialize cache directory and load existing metadata."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata: dict[str, Any] = self._load_metadata()

    def set_cache_dir(self, cache_dir: str | Path) -> None:
        """Change the cache directory and reload metadata."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        logger.info(f"Cache directory moved to: {self.cache_dir}")

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from the JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Metadata load failed: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Persist metadata to the disk."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Metadata save failed: {e}")

    def _get_path(self, key: str, data_type: type | None = None) -> Path:
        """Resolve the file path based on data type or metadata."""
        ext = ".dat"
        if data_type in self._EXTENSIONS:
            ext = self._EXTENSIONS[data_type]
        elif key in self.metadata:
            stored_type = self.metadata[key].get("data_type")
            ext = self._EXTENSIONS.get(self._TYPE_NAMES.get(stored_type, None), ".dat")
        return self.cache_dir / f"{key}{ext}"

    @staticmethod
    def _execute_save(data: Any, path: Path, dtype: type, **kwargs) -> dict:
        """Execute type-specific save operations."""
        if dtype is pd.DataFrame:
            data.to_csv(path, index=kwargs.get("index", False), encoding="utf-8")
            return {"row_count": len(data), "file_size": path.stat().st_size}

        if dtype in (dict, list):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return {"item_count": len(data), "file_size": path.stat().st_size}

        if dtype is str:
            path.write_text(data, encoding=kwargs.get("encoding", "utf-8"))
            return {"char_count": len(data), "file_size": path.stat().st_size}

        raise ValueError(f"Unsupported type: {dtype}")

    @staticmethod
    def _execute_load(path: Path, type_name: str, **kwargs) -> Any:
        """Execute type-specific load operations."""
        if type_name == "DataFrame":
            return pd.read_csv(path, encoding=kwargs.get("encoding", "utf-8"))
        if type_name in ("dict", "list"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if type_name == "str":
            return path.read_text(encoding=kwargs.get("encoding", "utf-8"))
        raise ValueError(f"Unknown data type in metadata: {type_name}")

    def save(self, key: str, data: Any, expire_hours: float | None = None, **kwargs) -> bool:
        """Save data to cache with optional expiration."""
        try:
            dtype = type(data)
            path = self._get_path(key, dtype)
            stats = self._execute_save(data, path, dtype, **kwargs)

            now = datetime.now()
            self.metadata[key] = {
                "created_at": now.isoformat(),
                "expire_at": (now + timedelta(hours=expire_hours)).isoformat() if expire_hours else None,
                "data_type": dtype.__name__,
                **stats,
            }
            self._save_metadata()
            return True
        except Exception as e:
            logger.error(f"Save failed for {key}: {e}")
            return False

    def load(self, key: str, auto_clean: bool = True, **kwargs) -> Any | None:
        """Load data from cache if not expired."""
        if self._is_expired(key):
            if auto_clean:
                self.delete(key)
            return None

        path = self._get_path(key)
        if not path.exists() or key not in self.metadata:
            return None

        try:
            return self._execute_load(path, self.metadata[key]["data_type"], **kwargs)
        except Exception as e:
            logger.error(f"Load failed for {key}: {e}")
            return None

    def _is_expired(self, key: str) -> bool:
        """Check if the cached entry has expired."""
        entry = self.metadata.get(key)
        if not entry or not entry.get("expire_at"):
            return False
        return datetime.now() > datetime.fromisoformat(entry["expire_at"])

    def delete(self, key: str) -> bool:
        """Remove a specific cache entry and its file."""
        try:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
            return True
        except OSError as e:
            logger.error(f"Delete failed for {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if a valid cache entry exists."""
        return key in self.metadata and not self._is_expired(key)

    def clear_all(self) -> bool:
        """Purge all cache files and reset metadata."""
        try:
            for file in self.cache_dir.iterdir():
                if file.is_file():
                    file.unlink()
            self.metadata = {}
            self._save_metadata()
            return True
        except OSError as e:
            logger.error(f"Clear all failed: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Return cache usage statistics."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())
        return {
            "count": len(self.metadata),
            "size_mb": round(total_size / (1024 * 1024), 2),
            "dir": str(self.cache_dir),
        }
