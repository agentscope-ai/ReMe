"""Prompt handler for loading and formatting prompt templates.

This module provides a PromptHandler class that manages prompt templates
with support for YAML file loading, language-specific prompts, and
conditional formatting.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .base_context import BaseContext
from .service_context import C


class PromptHandler(BaseContext):
    """Handler for loading, storing, and formatting prompt templates.

    This class extends BaseContext to provide prompt template management
    with support for:
    - Loading prompts from YAML files
    - Language-specific prompt retrieval
    - Conditional line filtering with boolean flags
    - Variable substitution in prompts

    Attributes:
        language: Language suffix for prompt retrieval (e.g., "zh", "en").

    Example:
        >>> handler = PromptHandler(language="zh")
        >>> handler.load_prompt_by_file("prompts.yaml")
        >>> prompt = handler.prompt_format("greeting", name="Alice", show_emoji=True)
    """

    def __init__(self, language: str = "", **kwargs):
        """Initialize PromptHandler with language setting.

        Args:
            language: Language suffix for prompt keys. If empty, uses
                the global language setting from ServiceContext.
            **kwargs: Additional context data to store.
        """
        super().__init__(**kwargs)
        self.language: str = language or C.language

    def load_prompt_by_file(
        self, prompt_file_path: Path | str | None = None, strict: bool = False
    ) -> "PromptHandler":
        """Load prompts from a YAML file.

        Args:
            prompt_file_path: Path to the YAML file containing prompts.
                If None, returns self without loading.
            strict: If True, raise FileNotFoundError when file doesn't exist.
                If False, log warning and return self.

        Returns:
            Self for method chaining.

        Raises:
            FileNotFoundError: If strict=True and file doesn't exist.
        """
        if prompt_file_path is None:
            return self

        if isinstance(prompt_file_path, str):
            prompt_file_path = Path(prompt_file_path)

        if not prompt_file_path.exists():
            if strict:
                raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
            logger.warning(f"Prompt file not found: {prompt_file_path}")
            return self

        with prompt_file_path.open(encoding="utf-8") as f:
            prompt_dict = yaml.load(f, yaml.FullLoader)
            self.load_prompt_dict(prompt_dict)
        return self

    def load_prompt_dict(self, prompt_dict: dict | None = None) -> "PromptHandler":
        """Load prompts from a dictionary.

        Args:
            prompt_dict: Dictionary mapping prompt names to prompt strings.
                Non-string values are ignored.

        Returns:
            Self for method chaining.
        """
        if not prompt_dict:
            return self

        for key, value in prompt_dict.items():
            if isinstance(value, str):
                if key in self:
                    self[key] = value
                    logger.warning(f"prompt_dict key={key} overwrite!")
                else:
                    self[key] = value
                    logger.debug(f"add prompt_dict key={key}")
        return self

    def _resolve_prompt_key(self, prompt_name: str) -> str:
        """Resolve the actual prompt key with language suffix.

        Args:
            prompt_name: Base prompt name.

        Returns:
            Resolved key with language suffix if applicable.
        """
        key = prompt_name
        if self.language and not key.endswith(self.language.strip()):
            key += "_" + self.language.strip()
        return key

    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists.

        Args:
            prompt_name: Name of the prompt to check.

        Returns:
            True if the prompt exists, False otherwise.
        """
        key = self._resolve_prompt_key(prompt_name)
        return key in self

    def get_prompt(self, prompt_name: str, default: Any = None) -> str:
        """Get a prompt by name.

        Args:
            prompt_name: Name of the prompt. Language suffix is automatically
                appended if language is set.
            default: Default value to return if prompt not found.
                If None and prompt not found, raises KeyError.

        Returns:
            The prompt string.

        Raises:
            KeyError: If prompt not found and no default provided.
        """
        key = self._resolve_prompt_key(prompt_name)

        if key not in self:
            if default is not None:
                return default
            raise KeyError(f"Prompt not found: {key}")
        return self[key]

    @staticmethod
    def _filter_conditional_lines(prompt: str, flag_kwargs: dict[str, bool]) -> str:
        """Filter prompt lines based on boolean flags.

        Lines starting with [flag_name] are conditionally included based on
        the corresponding boolean value in flag_kwargs.

        Args:
            prompt: The prompt string to filter.
            flag_kwargs: Dictionary of flag names to boolean values.

        Returns:
            Filtered prompt with conditional lines processed.
        """
        split_prompt = []
        for line in prompt.strip().split("\n"):
            hit = False
            hit_flag = True
            for key, flag in flag_kwargs.items():
                if not line.startswith(f"[{key}]"):
                    continue

                hit = True
                hit_flag = flag
                line = line.removeprefix(f"[{key}]")
                break

            if not hit:
                split_prompt.append(line)
            elif hit_flag:
                split_prompt.append(line)

        return "\n".join(split_prompt)

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Get and format a prompt with variable substitution.

        Supports two types of keyword arguments:
        - Boolean flags: Filter lines starting with [flag_name] based on the flag value.
        - Other values: Substituted into the prompt using str.format().

        Args:
            prompt_name: Name of the prompt to format.
            **kwargs: Variables for substitution. Boolean values are used for
                conditional line filtering.

        Returns:
            The formatted prompt string.

        Example:
            Given prompt "greeting":
                Hello {name}!
                [formal]It's a pleasure to meet you.
                [casual]What's up?

            >>> handler.prompt_format("greeting", name="Alice", formal=True, casual=False)
            "Hello Alice!\\nIt's a pleasure to meet you."
        """
        prompt = self.get_prompt(prompt_name)

        flag_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        other_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        if flag_kwargs:
            prompt = self._filter_conditional_lines(prompt, flag_kwargs)

        if other_kwargs:
            prompt = prompt.format(**other_kwargs)

        return prompt
