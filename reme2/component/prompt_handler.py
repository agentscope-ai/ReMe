"""Module for managing and formatting prompt templates."""

import inspect
import json
import re
from pathlib import Path
from string import Formatter

import yaml

_FLAG_PATTERN = re.compile(r"^\[(\w+)\]")


class PromptHandler:
    """A handler for loading, retrieving, and formatting prompt templates."""

    _SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}

    def __init__(self, language: str = "", **kwargs):
        self.data: dict[str, str] = {k: v for k, v in kwargs.items() if isinstance(v, str)}
        self.language: str = language.strip()

    def load_prompt_by_file(
            self,
            prompt_file_path: str | Path | None = None,
            overwrite: bool = True,
    ) -> "PromptHandler":
        """Load prompts from a YAML or JSON file."""
        if prompt_file_path is None:
            return self

        path = Path(prompt_file_path)
        if not path.exists() or path.suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            return self

        try:
            with path.open(encoding="utf-8") as f:
                prompt_dict = yaml.safe_load(f) if path.suffix.lower() in (".yaml", ".yml") else json.load(f)
        except (json.JSONDecodeError, yaml.YAMLError, OSError):
            return self

        return self.load_prompt_dict(prompt_dict, overwrite)

    def load_prompt_by_class(self, cls: type, overwrite: bool = True) -> "PromptHandler":
        """Load prompts from a YAML file named after the class."""
        try:
            base_path = Path(inspect.getfile(cls)).with_suffix("")
        except (TypeError, OSError):
            return self

        for ext in (".yaml", ".yml"):
            if (prompt_path := base_path.with_suffix(ext)).exists():
                return self.load_prompt_by_file(prompt_path, overwrite)

        return self

    def load_prompt_dict(self, prompt_dict: dict | None = None, overwrite: bool = True) -> "PromptHandler":
        """Merge prompts from a dictionary."""
        if not isinstance(prompt_dict, dict):
            return self

        for key, value in prompt_dict.items():
            if isinstance(value, str) and (overwrite or key not in self.data):
                self.data[key] = value

        return self

    def get_prompt(self, prompt_name: str) -> str:
        """Retrieve a prompt by name with language suffix fallback."""
        for key in (f"{prompt_name}_{self.language}", prompt_name) if self.language else (prompt_name,):
            if key in self.data:
                return self.data[key].strip()

        raise KeyError(f"Prompt '{prompt_name}' not found. Available: {list(self.data.keys())[:10]}")

    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists."""
        keys = (f"{prompt_name}_{self.language}", prompt_name) if self.language else (prompt_name,)
        return any(k in self.data for k in keys)

    def list_prompts(self, language_filter: str | None = None) -> list[str]:
        """List all available prompt names."""
        if language_filter is None:
            return list(self.data.keys())
        suffix = f"_{language_filter.strip()}"
        return [k for k in self.data if k.endswith(suffix)]

    def prompt_format(self, prompt_name: str, validate: bool = True, **kwargs) -> str:
        """Format a prompt with conditional line filtering and variable substitution."""
        prompt = self.get_prompt(prompt_name)
        flags = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        formats = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        if flags:
            lines = []
            for line in prompt.split("\n"):
                active_flags = _FLAG_PATTERN.findall(line)
                cleaned = _FLAG_PATTERN.sub("", line).lstrip()
                if not active_flags or any(flags.get(f, False) for f in active_flags):
                    lines.append(cleaned)
            prompt = "\n".join(lines)

        if validate:
            required = {f for _, f, _, _ in Formatter().parse(prompt) if f is not None}
            if missing := required - set(formats.keys()):
                raise ValueError(f"Missing format variables for '{prompt_name}': {sorted(missing)}")

        return prompt.format(**formats).strip() if formats else prompt

    def __repr__(self) -> str:
        return f"PromptHandler(language='{self.language}', num_prompts={len(self.data)})"
