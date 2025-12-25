"""Prompt handler for loading and managing prompts.

This module provides a class for loading prompts from files, managing
language-specific prompts, and formatting prompts with variables.
"""

from pathlib import Path

import yaml
from loguru import logger

from .base_context import BaseContext
from .service_context import C


class PromptHandler(BaseContext):
    """Handler for loading, storing, and formatting prompts.

    This class manages prompts loaded from YAML files, supports
    language-specific prompts, and provides formatting with variables
    and conditional flags.

    Attributes:
        language: Language code for language-specific prompts.
    """

    def __init__(self, language: str = "", **kwargs):
        """Initialize PromptHandler with language setting.

        Args:
            language: Language code for language-specific prompts.
                If not provided, uses the language from service context.
            **kwargs: Additional context data to store.
        """
        super().__init__(**kwargs)
        self.language: str = language or C.language

    def load_prompt_by_file(self, prompt_file_path: Path | str = None):
        """Load prompts from a YAML file.

        Args:
            prompt_file_path: Path to the YAML file containing prompts.
                Can be a Path object or string. If None, does nothing.

        Returns:
            Self for method chaining.
        """
        if prompt_file_path is None:
            return self

        if isinstance(prompt_file_path, str):
            prompt_file_path = Path(prompt_file_path)

        if not prompt_file_path.exists():
            return self

        with prompt_file_path.open(encoding="utf-8") as f:
            prompt_dict = yaml.load(f, yaml.FullLoader)
            self.load_prompt_dict(prompt_dict)
        return self

    def load_prompt_dict(self, prompt_dict: dict = None):
        """Load prompts from a dictionary.

        Args:
            prompt_dict: Dictionary of prompt names to prompt strings.
                If None, does nothing.

        Returns:
            Self for method chaining.
        """
        if not prompt_dict:
            return self

        for key, value in prompt_dict.items():
            if isinstance(value, str):
                if key in self._data:
                    self._data[key] = value
                    logger.warning(f"prompt_dict key={key} overwrite!")

                else:
                    self._data[key] = value
                    logger.debug(f"add prompt_dict key={key}")
        return self

    def get_prompt(self, prompt_name: str):
        """Get a prompt by name, with language suffix if applicable.

        Args:
            prompt_name: Base name of the prompt to retrieve.

        Returns:
            The prompt string.

        Raises:
            AssertionError: If the prompt (with language suffix) is not found.
        """
        key: str = prompt_name
        if self.language and not key.endswith(self.language.strip()):
            key += "_" + self.language.strip()

        assert key in self._data, f"prompt_name={key} not found."
        return self._data[key]

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Format a prompt with variables and conditional flags.

        This method supports two types of formatting:
        1. Boolean flags: Lines starting with [flag_name] are included
           only if the corresponding flag is True.
        2. Variable substitution: Other kwargs are used for string
           formatting with {variable_name}.

        Args:
            prompt_name: Name of the prompt to format.
            **kwargs: Variables and flags for formatting.

        Returns:
            The formatted prompt string.
        """
        prompt = self.get_prompt(prompt_name)

        flag_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        other_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        if flag_kwargs:
            split_prompt = []
            for line in prompt.strip().split("\n"):
                hit = False
                hit_flag = True
                for key, flag in kwargs.items():
                    if not line.startswith(f"[{key}]"):
                        continue

                    hit = True
                    hit_flag = flag
                    line = line.strip(f"[{key}]")
                    break

                if not hit:
                    split_prompt.append(line)
                elif hit_flag:
                    split_prompt.append(line)

            prompt = "\n".join(split_prompt)

        if other_kwargs:
            prompt = prompt.format(**other_kwargs)

        return prompt
