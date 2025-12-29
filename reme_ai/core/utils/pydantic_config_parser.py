"""Pydantic configuration parser for loading and merging configurations from multiple sources.

This module provides a generic parser that can load configurations from YAML files,
command-line arguments, and programmatic updates, with support for:
- Dot notation configuration keys (e.g., 'a.b.c=value')
- Deep merging of multiple configuration sources
- Automatic type conversion from strings
"""

import copy
import json
from pathlib import Path
from typing import Any, Generic, List, Type, TypeVar

import yaml
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PydanticConfigParser(Generic[T]):
    """Generic parser for Pydantic-based configuration management.

    This parser supports loading configurations from multiple sources with a priority order:
    1. Default values from Pydantic model
    2. YAML configuration files
    3. Command-line arguments (dot notation format)
    4. Programmatic updates via update_config()

    The parser automatically handles:
    - Type conversion (string to int, float, bool, None, JSON)
    - Deep merging of nested dictionaries
    - Dot notation for nested configuration keys

    Attributes:
        current_file: Path to the current file, used for relative config file resolution
        default_config_name: Default configuration file name (without .yaml extension)
        config_class: The Pydantic model class used for validation
        config_dict: The current merged configuration dictionary

    Example:
        ```python
        from pydantic import BaseModel, ConfigDict

        class AppConfig(BaseModel):
            model_config = ConfigDict(extra="allow")

            host: str = "localhost"
            port: int = 8080


        parser = PydanticConfigParser(AppConfig)
        # Override top-level fields directly
        config = parser.parse_args("config=app.yaml", "port=9000", "aaa=bbb")
        print(config.port)  # 9000
        print(config.aaa)  # bbb
        
        # Or use nested structure with dot notation
        config = parser.parse_args("config=app.yaml", "server.port=9000", "server.aaa=bbb")
        print(config.server['port'])  # 9000
        print(config.server['aaa'])  # bbb
        ```
    """

    current_file: str = __file__
    default_config_name: str = "default"

    def __init__(self, config_class: Type[T]):
        """Initialize configuration parser.

        Args:
            config_class: Pydantic configuration model class that defines the schema
                and default values for the configuration.

        Raises:
            TypeError: If config_class is not a subclass of BaseModel.
        """
        self.config_class = config_class
        self.config_dict: dict = {}

    def parse_dot_notation(self, dot_list: List[str]) -> dict:
        """Parse dot notation format configuration list into nested dictionary.

        Converts a list of dot-notation key-value pairs into a nested dictionary structure.
        Automatically performs type conversion on values.

        Args:
            dot_list: Configuration list in format ['a.b.c=value', 'x.y=123', 'z=true'].
                Items without '=' are ignored.

        Returns:
            Parsed nested dictionary. For example, ['a.b=1', 'a.c=2'] becomes
            {'a': {'b': 1, 'c': 2}}.

        Example:
            ```python
            parser = PydanticConfigParser(SomeConfig)
            result = parser.parse_dot_notation(['server.host=localhost', 'server.port=8080'])
            print(result)
            # {'server': {'host': 'localhost', 'port': 8080}}
            ```
        """
        config_dict = {}

        for item in dot_list:
            if "=" not in item:
                continue

            key_path, value_str = item.split("=", 1)
            keys = key_path.split(".")

            # Automatic type conversion
            value = self._convert_value(value_str)

            # Build nested dictionary
            current_dict = config_dict
            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]

            current_dict[keys[-1]] = value

        return config_dict

    @staticmethod
    def _convert_value(value_str: str) -> Any:
        """Automatically convert string values to appropriate Python types.

        Attempts to convert string values to their appropriate Python types:
        - Boolean: 'true'/'false' -> bool
        - None: 'none'/'null' -> None
        - Integer: Numeric strings without decimal point -> int
        - Float: Numeric strings with decimal point or scientific notation -> float
        - JSON: Valid JSON strings -> parsed JSON object
        - String: Everything else -> str

        Args:
            value_str: String value to convert.

        Returns:
            Converted value with appropriate Python type. If conversion fails,
            returns the original string.

        Example:
            ```python
            PydanticConfigParser._convert_value("123")  # 123
            PydanticConfigParser._convert_value("true")  # True
            PydanticConfigParser._convert_value('{"key": "value"}')  # {'key': 'value'}
            ```
        """
        value_str = value_str.strip()

        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        if value_str.lower() in ("none", "null"):
            return None

        try:
            if "." not in value_str and "e" not in value_str.lower():
                return int(value_str)

            return float(value_str)

        except ValueError:
            pass

        try:
            return json.loads(value_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return value_str

    @staticmethod
    def load_from_yaml(yaml_path: str | Path) -> dict:
        """Load configuration from YAML file.

        Reads and parses a YAML configuration file into a Python dictionary.

        Args:
            yaml_path: Path to the YAML configuration file. Can be a string or Path object.

        Returns:
            Configuration dictionary parsed from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist at the specified path.
            yaml.YAMLError: If the YAML file contains invalid YAML syntax.

        Example:
            ```python
            config = PydanticConfigParser.load_from_yaml("config.yaml")
            print(config)
            # {'host': 'localhost', 'port': 8080}
            ```
        """
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {yaml_path}")

        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def merge_configs(self, *config_dicts: dict) -> dict:
        """Deep merge multiple configuration dictionaries.

        Merges multiple configuration dictionaries in order, with later dictionaries
        overriding values from earlier ones. Performs deep merging for nested dictionaries,
        meaning nested values are merged rather than replaced entirely.

        Args:
            *config_dicts: Multiple configuration dictionaries to merge. Merging order
                is from first to last, with later configs taking precedence.

        Returns:
            Deeply merged configuration dictionary. All provided dictionaries are merged
            into a single result.

        Example:
            ```python
            parser = PydanticConfigParser(SomeConfig)
            base = {'a': {'x': 1, 'y': 2}}
            update = {'a': {'y': 3, 'z': 4}, 'b': 5}
            result = parser.merge_configs(base, update)
            print(result)
            # {'a': {'x': 1, 'y': 3, 'z': 4}, 'b': 5}
            ```
        """
        result = {}

        for config_dict in config_dicts:
            result = self._deep_merge(result, config_dict)

        return result

    def _deep_merge(self, base_dict: dict, update_dict: dict) -> dict:
        """Deep merge two dictionaries recursively.

        Recursively merges update_dict into base_dict. When both dictionaries contain
        the same key and both values are dictionaries, they are merged recursively.
        Otherwise, the value from update_dict overwrites the value in base_dict.

        Args:
            base_dict: Base dictionary to merge into. This dictionary is not modified.
            update_dict: Dictionary containing updates to merge into base_dict.

        Returns:
            New merged dictionary containing the combined result of both dictionaries.
            The base_dict and update_dict are not modified.

        Example:
            ```python
            parser = PydanticConfigParser(SomeConfig)
            base = {'a': 1, 'nested': {'x': 1}}
            update = {'b': 2, 'nested': {'y': 2}}
            result = parser._deep_merge(base, update)
            print(result)
            # {'a': 1, 'b': 2, 'nested': {'x': 1, 'y': 2}}
            ```
        """
        result = base_dict.copy()

        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def parse_args(self, *args) -> T:
        """Parse command line arguments and return validated configuration object.

        Parses configuration from multiple sources in priority order:
        1. Default values from Pydantic model
        2. YAML configuration file(s) (specified via config= or default_config_name)
            - Supports multiple configs separated by commas: config=default,search
            - Later configs override earlier ones
        3. Command-line overrides (dot notation format)

        Command-line arguments should be in format:
        - 'config=<filename>' or 'c=<filename>' to specify YAML config file
        - 'config=<file1>,<file2>' to specify multiple config files (file2 overrides file1)
        - 'key=value' or 'nested.key=value' for configuration overrides

        Args:
            *args: Command line arguments. Can include:
                - 'config=<file>' or 'c=<file>' to specify YAML config file
                - Dot notation overrides like 'server.host=localhost'

        Returns:
            Validated configuration object of type T (the Pydantic model class).
            The configuration is validated against the Pydantic schema.

        Raises:
            FileNotFoundError: If the specified config file does not exist.
            AssertionError: If no config file is specified and default_config_name is empty.
            ValidationError: If the merged configuration does not match the Pydantic schema.

        Example:
            ```python
            parser = PydanticConfigParser(AppConfig)
            config = parser.parse_args("config=app.yaml", "server.port=9000")
            print(config.port)  # 9000 (overridden from YAML default)

            # Multiple config files: search.yaml overrides default.yaml
            config = parser.parse_args("config=default,search")
            ```
        """
        configs_to_merge = []

        # 1. Default configuration (from Pydantic model)
        default_config = self.config_class().model_dump()
        configs_to_merge.append(default_config)

        # 2. YAML configuration file(s)
        config = ""
        filter_args = []
        for arg in args:
            if "=" not in arg:
                continue

            arg = arg.lstrip("--").lstrip("-")

            if "c=" in arg or "config=" in arg:
                config = arg.split("=")[-1]
            else:
                filter_args.append(arg)

        if not config:
            if self.default_config_name:
                config = self.default_config_name
            assert config, "add `config=<config_file>` in cmd!"

        # Support multiple config files separated by commas
        # Example: config=default,search (search will override default)
        config_list = [c.strip() for c in config.split(",") if c.strip()]

        # Load all config files in order (later configs override earlier ones)
        for single_config in config_list:
            if not single_config.endswith(".yaml"):
                single_config += ".yaml"

            # Resolve config file path
            config_path = Path(self.current_file).parent / single_config
            if config_path.exists():
                logger.info(f"load config={config_path}")
            else:
                logger.warning(f"config={config_path} not found, try {single_config}")
                config_path = Path(single_config)
                assert config_path.exists(), f"config={config_path} not found"

            yaml_config = self.load_from_yaml(config_path)
            configs_to_merge.append(yaml_config)

        # 3. Command line override configuration
        if args:
            cli_config = self.parse_dot_notation(filter_args)
            configs_to_merge.append(cli_config)

        # Merge all configurations
        self.config_dict = self.merge_configs(*configs_to_merge)

        # Create and validate final configuration object
        return self.config_class.model_validate(self.config_dict)

    def update_config(self, **kwargs) -> T:
        """Update configuration object using keyword arguments.

        Updates the current configuration with new values provided as keyword arguments.
        Supports dot notation via double underscores (e.g., server__host becomes server.host).

        Args:
            **kwargs: Configuration items to update. Keys can use double underscores
                (__) to represent nested paths (e.g., server__host='localhost' becomes
                server.host='localhost'). Values are automatically type-converted.

        Returns:
            Updated and validated configuration object of type T. The original
            config_dict is not modified.

        Example:
            ```python
            parser = PydanticConfigParser(AppConfig)
            config = parser.parse_args("config=app.yaml")
            updated = parser.update_config(server__port=9000, debug=True)
            print(updated.port)  # 9000
            print(updated.debug)  # True
            ```
        """
        # Convert kwargs to dot notation format
        dot_list = []
        for key, value in kwargs.items():
            # support double underscore as dot replacement (server__host -> server.host)
            dot_key = key.replace("__", ".")
            dot_list.append(f"{dot_key}={value}")

        # Parse and merge configuration
        override_config = self.parse_dot_notation(dot_list)
        final_config = self.merge_configs(copy.deepcopy(self.config_dict), override_config)

        return self.config_class.model_validate(final_config)
