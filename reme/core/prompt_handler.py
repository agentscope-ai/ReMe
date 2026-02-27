"""Module for managing and formatting prompt templates from files or dictionaries."""

import json
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Optional, Union

import yaml
from loguru import logger

from .base_dict import BaseDict


class PromptHandler(BaseDict):
    """A context-aware handler for loading, retrieving, and formatting prompt templates."""

    def __init__(self, language: str = "", **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ to avoid storing 'language' in the dict
        object.__setattr__(self, "language", language.strip())

    def load_prompt_by_file(
        self,
        prompt_file_path: Optional[Union[Path, str]] = None,
        overwrite: bool = True,
    ) -> "PromptHandler":
        """Load prompt configurations from a YAML or JSON file."""
        if prompt_file_path is None:
            return self

        if isinstance(prompt_file_path, str):
            prompt_file_path = Path(prompt_file_path)

        if not prompt_file_path.exists():
            return self

        suffix = prompt_file_path.suffix.lower()

        with prompt_file_path.open(encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                prompt_dict = yaml.safe_load(f)
            elif suffix == ".json":
                prompt_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        self.load_prompt_dict(prompt_dict, overwrite=overwrite)
        return self

    def load_prompt_dict(
        self,
        prompt_dict: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
    ) -> "PromptHandler":
        """Merge a dictionary of prompt strings into the current context."""
        if not prompt_dict:
            return self

        for key, value in prompt_dict.items():
            if not isinstance(value, str):
                continue
            if key in self:
                if overwrite:
                    logger.warning(f"Overwriting prompt '{key}'")
                    self[key] = value
            else:
                self[key] = value

        return self

    def get_prompt(self, prompt_name: str, fallback_to_base: bool = True) -> str:
        """Retrieve a prompt by name with automatic language suffix handling."""
        if self.language and not prompt_name.endswith(f"_{self.language}"):
            key_with_lang = f"{prompt_name}_{self.language}"
            if key_with_lang in self:
                return self[key_with_lang].strip()

        if prompt_name in self:
            return self[prompt_name].strip()

        if fallback_to_base and self.language and prompt_name.endswith(f"_{self.language}"):
            base_name = prompt_name[: -(len(self.language) + 1)]
            if base_name in self:
                return self[base_name].strip()

        raise KeyError(f"Prompt '{prompt_name}' not found. Available: {list(self.keys())[:10]}")

    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists."""
        try:
            self.get_prompt(prompt_name)
            return True
        except KeyError:
            return False

    def list_prompts(self, language_filter: Optional[str] = None) -> list[str]:
        """List all available prompt names."""
        if language_filter is None:
            return list(self.keys())
        suffix = f"_{language_filter.strip()}"
        return [key for key in self.keys() if key.endswith(suffix)]

    @staticmethod
    def _extract_format_fields(template: str) -> set[str]:
        """Extract all format field names from a template string."""
        return {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None}

    @staticmethod
    def _filter_conditional_lines(prompt: str, flags: Dict[str, bool]) -> str:
        """Filter lines based on boolean flags."""
        filtered_lines = []
        for line in prompt.split("\n"):
            matched_flag = None
            for flag_name in flags:
                if line.startswith(f"[{flag_name}]"):
                    matched_flag = flag_name
                    break
            if matched_flag is None:
                filtered_lines.append(line)
            elif flags[matched_flag]:
                filtered_lines.append(line[len(f"[{matched_flag}]") :])
        return "\n".join(filtered_lines)

    def prompt_format(self, prompt_name: str, validate: bool = True, **kwargs) -> str:
        """Format a prompt with conditional line filtering and variable substitution."""
        prompt = self.get_prompt(prompt_name)

        flag_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        format_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        if flag_kwargs:
            prompt = self._filter_conditional_lines(prompt, flag_kwargs)

        if validate:
            required_fields = self._extract_format_fields(prompt)
            missing_fields = required_fields - set(format_kwargs.keys())
            if missing_fields:
                raise ValueError(f"Missing format variables for '{prompt_name}': {sorted(missing_fields)}")

        if format_kwargs:
            prompt = prompt.format(**format_kwargs)

        return prompt.strip()

    def __repr__(self) -> str:
        return f"PromptHandler(language='{self.language}', num_prompts={len(self)})"
