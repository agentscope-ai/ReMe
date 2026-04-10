"""Module for managing and formatting prompt templates."""

import json
from pathlib import Path
from string import Formatter

import yaml

from ..utils import get_logger


class PromptHandler:
    """A handler for loading, retrieving, and formatting prompt templates."""

    _SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}

    def __init__(self, language: str = "", **kwargs):
        self.data: dict[str, str] = {k: v for k, v in kwargs.items() if isinstance(v, str)}
        self.language: str = language.strip()
        self.logger = get_logger()

    def load_prompt_by_file(self, prompt_file_path: str | Path | None = None,
                            overwrite: bool = True) -> "PromptHandler":
        """Load prompts from a YAML or JSON file."""
        if prompt_file_path is None:
            return self

        path = Path(prompt_file_path)
        if not path.exists():
            self.logger.warning(f"Prompt file not found: {path}")
            return self

        suffix = path.suffix.lower()
        if suffix not in self._SUPPORTED_EXTENSIONS:
            self.logger.warning(f"Unsupported file extension '{suffix}', expected one of {self._SUPPORTED_EXTENSIONS}")
            return self

        try:
            with path.open(encoding="utf-8") as f:
                prompt_dict = yaml.safe_load(f) if suffix in (".yaml", ".yml") else json.load(f)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error(f"Failed to parse prompt file {path}: {e}")
            return self
        except OSError as e:
            self.logger.error(f"Failed to read prompt file {path}: {e}")
            return self

        return self.load_prompt_dict(prompt_dict, overwrite)

    def load_prompt_dict(self, prompt_dict: dict | None = None, overwrite: bool = True) -> "PromptHandler":
        """Merge prompts from a dictionary."""
        if not prompt_dict:
            return self

        for key, value in prompt_dict.items():
            if not isinstance(value, str):
                continue
            if key in self.data and not overwrite:
                continue
            if key in self.data:
                self.logger.warning(f"Overwriting prompt '{key}'")
            self.data[key] = value

        return self

    def get_prompt(self, prompt_name: str) -> str:
        """Retrieve a prompt by name with language suffix fallback."""
        if self.language:
            key = f"{prompt_name}_{self.language}"
            if key in self.data:
                return self.data[key].strip()

        if prompt_name in self.data:
            return self.data[prompt_name].strip()

        raise KeyError(f"Prompt '{prompt_name}' not found. Available: {list(self.data.keys())[:10]}")

    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists."""
        return prompt_name in self.data or f"{prompt_name}_{self.language}" in self.data

    def list_prompts(self, language_filter: str | None = None) -> list[str]:
        """List all available prompt names."""
        if not language_filter:
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
                remaining = line
                has_flag = False
                should_include = False
                while True:
                    matched = False
                    for flag, enabled in flags.items():
                        prefix = f"[{flag}]"
                        if remaining.startswith(prefix):
                            remaining = remaining[len(prefix):]
                            has_flag = True
                            if enabled:
                                should_include = True
                            matched = True
                            break
                    if not matched:
                        break
                # Include line if: no flag prefix, or at least one flag is enabled
                if not has_flag or should_include:
                    lines.append(remaining)
            prompt = "\n".join(lines)

        if validate:
            required = {f for _, f, _, _ in Formatter().parse(prompt) if f}
            missing = required - set(formats.keys())
            if missing:
                raise ValueError(f"Missing format variables for '{prompt_name}': {sorted(missing)}")

        return prompt.format(**formats).strip() if formats else prompt.strip()

    def __repr__(self) -> str:
        return f"PromptHandler(language='{self.language}', num_prompts={len(self.data)})"
