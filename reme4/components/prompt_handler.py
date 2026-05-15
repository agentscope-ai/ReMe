"""Prompt template loader and formatter with conditional-line and i18n support."""

import inspect
import json
import re
from pathlib import Path
from string import Formatter

import yaml

# Matches a leading flag tag like "[verbose] some text".
_FLAG_PATTERN = re.compile(r"^\[(\w+)]")


class PromptHandler:
    """Loads prompts from YAML/JSON or class-adjacent files and formats them.

    Templates may carry a language suffix (``key_en``, ``key_zh``); ``get_prompt``
    falls back to the bare key when no localized variant exists. ``prompt_format``
    additionally supports per-line flags such as ``[verbose] extra text`` that
    are kept only when the matching flag kwarg is truthy.
    """

    _SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}

    def __init__(self, language: str = "", **kwargs):
        # Only string entries are treated as prompts; other kwargs are ignored.
        self.data: dict[str, str] = {k: v for k, v in kwargs.items() if isinstance(v, str)}
        self.language: str = language.strip()

    def load_prompt_by_file(
        self,
        prompt_file_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> "PromptHandler":
        """Load prompts from a YAML or JSON file; silently skip on any error."""
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
        """Load prompts from ``<class_module>.yaml`` (or ``.yml``) next to `cls`."""
        try:
            base_path = Path(inspect.getfile(cls)).with_suffix("")
        except (TypeError, OSError):
            return self

        for ext in (".yaml", ".yml"):
            if (prompt_path := base_path.with_suffix(ext)).exists():
                return self.load_prompt_by_file(prompt_path, overwrite)

        return self

    def load_prompt_dict(self, prompt_dict: dict | None = None, overwrite: bool = True) -> "PromptHandler":
        """Merge string entries from `prompt_dict` into the in-memory store."""
        if not isinstance(prompt_dict, dict):
            return self

        for key, value in prompt_dict.items():
            if isinstance(value, str) and (overwrite or key not in self.data):
                self.data[key] = value

        return self

    def get_prompt(self, prompt_name: str) -> str:
        """Return the template, preferring the language-suffixed variant when set."""
        for key in (f"{prompt_name}_{self.language}", prompt_name) if self.language else (prompt_name,):
            if key in self.data:
                return self.data[key].strip()

        raise KeyError(f"Prompt '{prompt_name}' not found. Available: {list(self.data.keys())[:10]}")

    def has_prompt(self, prompt_name: str) -> bool:
        """True if either the localized or bare prompt is registered."""
        keys = (f"{prompt_name}_{self.language}", prompt_name) if self.language else (prompt_name,)
        return any(k in self.data for k in keys)

    def list_prompts(self, language_filter: str | None = None) -> list[str]:
        """List all keys, optionally filtered to those ending with ``_<language>``."""
        if language_filter is None:
            return list(self.data.keys())
        suffix = f"_{language_filter.strip()}"
        return [k for k in self.data if k.endswith(suffix)]

    def prompt_format(self, prompt_name: str, validate: bool = True, **kwargs) -> str:
        """Render a prompt: strip inactive flag-lines, then ``str.format`` it.

        Boolean kwargs are treated as flags controlling ``[flag]`` line filtering.
        Remaining kwargs become positional substitutions for ``{var}`` placeholders.
        With `validate=True`, missing substitutions raise ``ValueError``.
        """
        prompt = self.get_prompt(prompt_name)
        flags = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        formats = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        # Keep lines without flags; otherwise keep when at least one flag is enabled.
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
