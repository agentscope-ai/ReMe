"""
CacheHandler utility that supports DataFrame, list, string,
and dict with local storage and data expiration functionality
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union

import pandas as pd
from loguru import logger


class CacheHandler:
    """
    Cache utility class supporting DataFrame, list, string, and dict

    Features:
    - Support for DataFrame (CSV), list (JSON), string (TXT), and dict (JSON)
    - Support for data expiration time settings
    - Automatic cleanup of expired data
    - Recording and managing update timestamps
    """

    def __init__(self, cache_dir: Union[str, Path] = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = {}
        self._load_metadata()

    def set_cache_dir(self, cache_dir: Union[str, Path]):
        """
        Set a new cache directory and reload metadata

        Args:
            cache_dir: New cache directory path
        """
        new_cache_dir = Path(cache_dir)
        new_cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = new_cache_dir
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = {}
        self._load_metadata()

        logger.info(f"Cache directory changed to: {self.cache_dir}")

    def _load_metadata(self):
        """Load metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = {}

    def _save_metadata(self):
        """Save metadata"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    @staticmethod
    def _get_file_extension(data_type: type) -> str:
        """Get file extension for data type"""
        if data_type is pd.DataFrame:
            return ".csv"
        elif data_type == dict:
            return ".json"
        elif data_type == list:
            return ".json"
        elif data_type == str:
            return ".txt"
        else:
            return ".dat"

    def _get_file_path(self, key: str, data_type: type = None) -> Path:
        """Get data file path with appropriate extension"""
        if data_type is None:
            # Try to get extension from metadata
            if key in self.metadata and "data_type" in self.metadata[key]:
                stored_type_name = self.metadata[key]["data_type"]
                if stored_type_name == "DataFrame":
                    extension = ".csv"
                elif stored_type_name == "dict":
                    extension = ".json"
                elif stored_type_name == "list":
                    extension = ".json"
                elif stored_type_name == "str":
                    extension = ".txt"
                else:
                    extension = ".dat"
            else:
                extension = ".dat"
        else:
            extension = self._get_file_extension(data_type)

        return self.cache_dir / f"{key}{extension}"

    @staticmethod
    def _save_dataframe(data: pd.DataFrame, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Save DataFrame as CSV"""
        csv_params = {
            "index": False,
            "encoding": "utf-8",
        }
        csv_params.update(kwargs)
        data.to_csv(file_path, **csv_params)
        return {
            "row_count": len(data),
            "column_count": len(data.columns),
            "file_size": file_path.stat().st_size,
        }

    @staticmethod
    def _load_dataframe(file_path: Path, **kwargs) -> pd.DataFrame:
        """Load DataFrame from CSV"""
        csv_params = {
            "encoding": "utf-8",
        }
        csv_params.update(kwargs)
        return pd.read_csv(file_path, **csv_params)

    @staticmethod
    def _save_dict(data: dict, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Save dict as JSON"""
        json_params = {
            "ensure_ascii": False,
            "indent": 2,
            **kwargs,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_params)
        return {
            "key_count": len(data),
            "file_size": file_path.stat().st_size,
        }

    @staticmethod
    def _load_dict(file_path: Path, **kwargs) -> dict:
        """Load dict from JSON"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f, **kwargs)

    @staticmethod
    def _save_list(data: list, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Save list as JSON"""
        json_params = {
            "ensure_ascii": False,
            "indent": 2,
        }
        json_params.update(kwargs)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_params)
        return {
            "item_count": len(data),
            "file_size": file_path.stat().st_size,
        }

    @staticmethod
    def _load_list(file_path: Path, **kwargs) -> list:
        """Load list from JSON"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f, **kwargs)

    @staticmethod
    def _save_string(data: str, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Save string as TXT"""
        encoding = kwargs.get("encoding", "utf-8")
        with open(file_path, "w", encoding=encoding) as f:
            f.write(data)
        return {
            "char_count": len(data),
            "file_size": file_path.stat().st_size,
        }

    @staticmethod
    def _load_string(file_path: Path, **kwargs) -> str:
        """Load string from TXT"""
        encoding = kwargs.get("encoding", "utf-8")
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()

    def _save_data(self, data: Any, file_path: Path, data_type: type, **kwargs) -> Dict[str, Any]:
        """Save data based on type"""
        if data_type is pd.DataFrame:
            return self._save_dataframe(data, file_path, **kwargs)
        elif data_type == dict:
            return self._save_dict(data, file_path, **kwargs)
        elif data_type == list:
            return self._save_list(data, file_path, **kwargs)
        elif data_type == str:
            return self._save_string(data, file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def _load_data(self, file_path: Path, data_type_name: str, **kwargs) -> Any:
        """Load data based on type name"""
        if data_type_name == "DataFrame":
            return self._load_dataframe(file_path, **kwargs)
        elif data_type_name == "dict":
            return self._load_dict(file_path, **kwargs)
        elif data_type_name == "list":
            return self._load_list(file_path, **kwargs)
        elif data_type_name == "str":
            return self._load_string(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {data_type_name}")

    def _is_expired(self, key: str) -> bool:
        """Check if data is expired"""
        if key not in self.metadata:
            return True

        expire_time_str = self.metadata[key].get("expire_time")
        if not expire_time_str:
            return False  # No expiration time set, never expires

        expire_time = datetime.fromisoformat(expire_time_str)
        return datetime.now() > expire_time

    def save(
        self,
        key: str,
        data: Union[pd.DataFrame, dict, list, str],
        expire_hours: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        Save data to cache

        Args:
            key: Cache key name
            data: Data to save (DataFrame, dict, list, or str)
            expire_hours: Expiration time in hours, None means never expires
            **kwargs: Additional parameters for save operations (e.g., encoding for string)

        Returns:
            bool: Whether save was successful
        """
        try:
            data_type = type(data)

            # Validate data type
            if data_type not in [pd.DataFrame, dict, list, str]:
                logger.error(f"Unsupported data type: {data_type}")
                return False

            file_path = self._get_file_path(key, data_type)

            # Save data
            handler_metadata = self._save_data(data, file_path, data_type, **kwargs)

            # Update metadata
            current_time = datetime.now()
            self.metadata[key] = {
                "created_time": current_time.isoformat(),
                "updated_time": current_time.isoformat(),
                "expire_time": ((current_time + timedelta(hours=expire_hours)).isoformat() if expire_hours else None),
                "data_type": data_type.__name__,
                **handler_metadata,
            }

            self._save_metadata()
            return True

        except Exception as e:
            logger.exception(f"Failed to save data: {e}")
            return False

    def load(
        self,
        key: str,
        auto_clean_expired: bool = True,
        **kwargs,
    ) -> Optional[Any]:
        """
        Load data from cache

        Args:
            key: Cache key name
            auto_clean_expired: Whether to automatically clean expired data
            **kwargs: Additional parameters for load operations (e.g., encoding for string)

        Returns:
            Optional[Any]: Loaded data, returns None if not exists or expired
        """
        try:
            # Check if expired
            if self._is_expired(key):
                if auto_clean_expired:
                    self.delete(key)
                return None

            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None

            # Get data type from metadata
            if key not in self.metadata or "data_type" not in self.metadata[key]:
                logger.warning(f"No data type information found for key '{key}'")
                return None

            data_type_name = self.metadata[key]["data_type"]

            # Load data
            data = self._load_data(file_path, data_type_name, **kwargs)

            # Update last access time
            if key in self.metadata:
                self.metadata[key]["last_accessed"] = datetime.now().isoformat()
                self._save_metadata()

            return data

        except Exception as e:
            logger.exception(f"Failed to load data: {e}")
            return None

    def exists(self, key: str, check_expired: bool = True) -> bool:
        """
        Check if cache exists

        Args:
            key: Cache key name
            check_expired: Whether to check expiration status

        Returns:
            bool: Whether cache exists and is not expired
        """
        if check_expired and self._is_expired(key):
            return False

        file_path = self._get_file_path(key)
        return file_path.exists() and key in self.metadata

    def delete(self, key: str) -> bool:
        """
        Delete cache

        Args:
            key: Cache key name

        Returns:
            bool: Whether deletion was successful
        """
        try:
            file_path = self._get_file_path(key)

            # Delete data file
            if file_path.exists():
                file_path.unlink()

            # Delete metadata
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()

            return True

        except Exception as e:
            logger.exception(f"Failed to delete cache: {e}")
            return False

    def clean_expired(self) -> int:
        """
        Clean all expired caches

        Returns:
            int: Number of cleaned caches
        """
        expired_keys = []

        for key in list(self.metadata.keys()):
            if self._is_expired(key):
                expired_keys.append(key)

        cleaned_count = 0
        for key in expired_keys:
            if self.delete(key):
                cleaned_count += 1

        return cleaned_count

    def get_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cache information

        Args:
            key: Cache key name

        Returns:
            Optional[Dict]: Cache information including creation time, update time, expiration time, etc.
        """
        if key not in self.metadata:
            return None

        info = self.metadata[key].copy()
        info["key"] = key
        info["is_expired"] = self._is_expired(key)
        info["file_path"] = str(self._get_file_path(key))

        return info

    def list_all(self, include_expired: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        List all caches

        Args:
            include_expired: Whether to include expired caches

        Returns:
            Dict: Information of all caches
        """
        result = {}

        for key in self.metadata:
            if not include_expired and self._is_expired(key):
                continue

            info = self.get_info(key)
            if info:
                result[key] = info

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict: Cache statistics information
        """
        total_count = len(self.metadata)
        expired_count = sum(1 for key in self.metadata if self._is_expired(key))
        active_count = total_count - expired_count

        total_size = 0
        for key in self.metadata:
            file_path = self._get_file_path(key)
            if file_path.exists():
                total_size += file_path.stat().st_size

        return {
            "total_count": total_count,
            "active_count": active_count,
            "expired_count": expired_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def clear_all(self) -> bool:
        """
        Clear all caches

        Returns:
            bool: Whether clearing was successful
        """
        try:
            # Delete all data files (CSV, JSON, TXT, and other supported formats)
            for data_file in self.cache_dir.glob("*"):
                if data_file.is_file() and data_file.name != "metadata.json":
                    data_file.unlink()

            # Clear metadata
            self.metadata = {}
            self._save_metadata()

            return True

        except Exception as e:
            logger.exception(f"Failed to clear cache: {e}")
            return False
