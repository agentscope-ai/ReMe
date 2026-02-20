"""Parser for Pydantic config models with YAML and CLI argument support."""

import inspect
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PydanticConfigParser:
    """Parser that loads and merges Pydantic configs from YAML files and CLI args."""

    def __init__(self, config_class: type[T], default_config: str = "default"):
        """Initialize parser with a Pydantic config class.

        Args:
            config_class: Pydantic BaseModel class to validate configs against.
            default_config: Default config file name to use if not specified in args.
        """
        self.config_class = config_class
        self.default_config = default_config
        self.config_dict: dict = {}

    def _deep_merge(self, base_dict: dict, update_dict: dict) -> dict:
        """Recursively merge two dictionaries."""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _convert_value(value_str: str) -> Any:
        """Convert string value to appropriate Python type."""
        value_str = value_str.strip()
        lower_str = value_str.lower()

        # Boolean and None conversion
        if lower_str in ("true", "false"):
            return lower_str == "true"
        if lower_str in ("none", "null"):
            return None

        # Numeric conversion
        if "e" in lower_str or "." in value_str:
            try:
                return float(value_str)
            except ValueError:
                pass
        else:
            try:
                return int(value_str)
            except ValueError:
                pass

        # JSON conversion for complex types
        try:
            return json.loads(value_str)
        except (json.JSONDecodeError, ValueError):
            return value_str

    @staticmethod
    def load_from_yaml(yaml_path: str | Path) -> dict:
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Dictionary containing configuration data.

        Raises:
            FileNotFoundError: If YAML file does not exist.
        """
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {yaml_path}")

        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def merge_configs(self, *config_dicts: dict) -> dict:
        """Merge multiple config dictionaries in order.

        Args:
            *config_dicts: Variable number of config dictionaries to merge.

        Returns:
            Merged configuration dictionary.
        """
        result = {}
        for config_dict in config_dicts:
            result = self._deep_merge(result, config_dict)
        return result

    def parse_dot_notation(self, dot_list: list[str]) -> dict:
        """Parse dot notation strings into nested dictionary.

        Args:
            dot_list: List of strings in format "key.subkey=value".

        Returns:
            Nested dictionary representation of dot notation.
        """
        config_dict = {}
        for item in dot_list:
            if "=" not in item:
                continue

            key_path, value_str = item.split("=", 1)
            keys = key_path.split(".")

            # Build nested dictionary
            current = config_dict
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = self._convert_value(value_str)

        return config_dict

    def _find_config_path(self, config_name: str) -> Path:
        """Find config file path, trying parser directory first then current directory."""
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        # Try parser class directory first
        config_path = Path(inspect.getfile(self.__class__)).parent / config_name
        if config_path.exists():
            logger.info(f"load config={config_path}")
            return config_path

        # Try current directory
        logger.warning(f"config={config_path} not found, try {config_name}")
        config_path = Path(config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"config={config_path} not found")
        return config_path

    def parse_args(self, *args: str, **kwargs) -> T:
        """Parse CLI arguments and load configs from YAML files."""
        configs_to_merge = [self.config_class().model_dump()]

        # Separate config file path from other arguments
        config = ""
        filter_args = []
        for arg in args:
            if "=" not in arg:
                continue
            arg = arg.lstrip("-")
            if arg.startswith(("c=", "config=")):
                config = arg.split("=", 1)[1]
            else:
                filter_args.append(arg)

        # Use default config if not specified
        config = config or self.default_config

        # Load each config file
        for single_config in (c.strip() for c in config.split(",") if c.strip()):
            config_path = self._find_config_path(single_config)
            configs_to_merge.append(self.load_from_yaml(config_path))

        # Apply CLI overrides
        if filter_args:
            configs_to_merge.append(self.parse_dot_notation(filter_args))

        if kwargs:
            configs_to_merge.append(kwargs)

        # Merge all configs and validate
        self.config_dict = self.merge_configs(*configs_to_merge)
        return self.config_class.model_validate(self.config_dict, extra="allow")

    def update_config(self, **kwargs) -> T:
        """Update current config with new values using kwargs.

        Args:
            **kwargs: Key-value pairs where __ in keys represents nested levels.

        Returns:
            Updated and validated Pydantic config instance.
        """
        # Convert kwargs to dot notation and parse
        dot_list = [f"{key.replace('__', '.')}={value}" for key, value in kwargs.items()]
        override_config = self.parse_dot_notation(dot_list)

        # Merge with existing config
        final_config = self.merge_configs(self.config_dict, override_config)
        return self.config_class.model_validate(final_config, extra="allow")
