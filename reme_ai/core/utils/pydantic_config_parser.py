import copy
import inspect
import json
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PydanticConfigParser:
    default_config: str = "default"

    def __init__(self, config_class: Type[T]):
        self.config_class = config_class
        self.config_dict: dict = {}

    def _deep_merge(self, base_dict: dict, update_dict: dict) -> dict:
        result = copy.deepcopy(base_dict)

        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    @staticmethod
    def _convert_value(value_str: str) -> Any:
        value_str = value_str.strip()

        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        if value_str.lower() in ("none", "null"):
            return None

        try:
            lower_str = value_str.lower()
            if "e" in lower_str or "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        try:
            return json.loads(value_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return value_str

    @staticmethod
    def load_from_yaml(yaml_path: str | Path) -> dict:
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {yaml_path}")

        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def merge_configs(self, *config_dicts: dict) -> dict:
        result = {}

        for config_dict in config_dicts:
            result = self._deep_merge(result, config_dict)

        return result

    def parse_dot_notation(self, dot_list: list[str]) -> dict:
        config_dict = {}

        for item in dot_list:
            if "=" not in item:
                continue

            key_path, value_str = item.split("=", 1)
            keys = key_path.split(".")
            value = self._convert_value(value_str)
            current_dict = config_dict
            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]

            current_dict[keys[-1]] = value

        return config_dict

    def parse_args(self, *args: str) -> T:
        configs_to_merge = [self.config_class().model_dump()]

        config = ""
        filter_args = []
        for arg in args:
            if "=" not in arg:
                continue

            arg = arg.lstrip("-")

            if arg.startswith("c=") or arg.startswith("config="):
                config = arg.split("=", 1)[-1]
            else:
                filter_args.append(arg)

        if not config:
            config = self.default_config
        if not config:
            raise ValueError("add `config=<config_file>` in cmd!")

        config_list = [c.strip() for c in config.split(",") if c.strip()]

        for single_config in config_list:
            if not single_config.endswith(".yaml"):
                single_config += ".yaml"

            config_path = Path(inspect.getfile(self.__class__)).parent / single_config
            if config_path.exists():
                logger.info(f"load config={config_path}")
            else:
                logger.warning(f"config={config_path} not found, try {single_config}")
                config_path = Path(single_config)
                if not config_path.exists():
                    raise FileNotFoundError(f"config={config_path} not found")

            yaml_config = self.load_from_yaml(config_path)
            configs_to_merge.append(yaml_config)

        if filter_args:
            cli_config = self.parse_dot_notation(filter_args)
            configs_to_merge.append(cli_config)

        self.config_dict = self.merge_configs(*configs_to_merge)
        return self.config_class.model_validate(self.config_dict)

    def update_config(self, **kwargs) -> T:
        dot_list = []
        for key, value in kwargs.items():
            dot_key = key.replace("__", ".")
            dot_list.append(f"{dot_key}={value}")

        override_config = self.parse_dot_notation(dot_list)
        final_config = self.merge_configs(copy.deepcopy(self.config_dict), override_config)

        return self.config_class.model_validate(final_config)
