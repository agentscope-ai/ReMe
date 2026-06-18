"""Prompt template loader and formatter with conditional-line and i18n support."""

import inspect
import json
import re
from pathlib import Path

import yaml

# Matches a leading flag tag at line start: "[flag] rest of line".
_FLAG_PATTERN = re.compile(r"^\[(\w+)]")


class PromptHandler:
    """Loads prompts from YAML/JSON files and renders them with optional flags.

    Template keys may carry a language suffix (``key_en``, ``key_zh``); lookups
    fall back to the bare key when no localized variant exists. Lines tagged
    with ``[flag]`` are kept only when the matching boolean kwarg is truthy.
    """

    _SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}

    def __init__(self, language: str = "", **kwargs):
        # Non-string kwargs are silently dropped — prompts must be strings.
        self.data: dict[str, str] = {k: v for k, v in kwargs.items() if isinstance(v, str)}
        self.language: str = language.strip()

    # ----- Loading -------------------------------------------------------

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

        return self.load_prompt_dict(self._parse_prompt_file(path), overwrite)

    @staticmethod
    def _parse_prompt_file(path: Path) -> dict | None:
        """Parse a YAML or JSON prompt file; return None on any parse error."""
        try:
            with path.open(encoding="utf-8") as f:
                if path.suffix.lower() in (".yaml", ".yml"):
                    return yaml.safe_load(f)
                return json.load(f)
        except (json.JSONDecodeError, yaml.YAMLError, OSError):
            return None

    def load_prompt_by_class(self, cls: type, overwrite: bool = True) -> "PromptHandler":
        """Load prompts from ``<class_module>.yaml`` (or ``.yml``) next to `cls`."""
        try:
            base_path = Path(inspect.getfile(cls)).with_suffix("")
        except (TypeError, OSError):
            return self

        for ext in (".yaml", ".yml"):
            candidate = base_path.with_suffix(ext)
            if candidate.exists():
                return self.load_prompt_by_file(candidate, overwrite)
        return self

    def load_prompt_dict(self, prompt_dict: dict | None = None, overwrite: bool = True) -> "PromptHandler":
        """Merge string entries from `prompt_dict` into the in-memory store."""
        if not isinstance(prompt_dict, dict):
            return self

        for key, value in prompt_dict.items():
            if isinstance(value, str) and (overwrite or key not in self.data):
                self.data[key] = value
        return self

    # ----- Lookup --------------------------------------------------------

    def _candidate_keys(self, prompt_name: str) -> tuple[str, ...]:
        """Lookup order: localized key first when a language is set, then bare key."""
        if self.language:
            return (f"{prompt_name}_{self.language}", prompt_name)
        return (prompt_name,)

    def get_prompt(self, prompt_name: str) -> str:
        """Return the template, preferring the language-suffixed variant when set."""
        for key in self._candidate_keys(prompt_name):
            if key in self.data:
                return self.data[key].strip()
        raise KeyError(
            f"Prompt '{prompt_name}' not found. Available: {list(self.data.keys())[:10]}",
        )

    def has_prompt(self, prompt_name: str) -> bool:
        """True if either the localized or bare prompt is registered."""
        return any(k in self.data for k in self._candidate_keys(prompt_name))

    def list_prompts(self, language_filter: str | None = None) -> list[str]:
        """List all keys, optionally filtered to those ending with ``_<language>``."""
        if language_filter is None:
            return list(self.data.keys())
        suffix = f"_{language_filter.strip()}"
        return [k for k in self.data if k.endswith(suffix)]

    # ----- Formatting ----------------------------------------------------

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Render a prompt: strip inactive ``[flag]`` lines, then ``str.format`` it.

        Boolean kwargs are treated as flag toggles; the rest are format variables.
        """
        prompt = self.get_prompt(prompt_name)
        flags = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        formats = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        if flags:
            prompt = self._apply_flag_filter(prompt, flags)

        return prompt.format(**formats).strip() if formats else prompt

    @staticmethod
    def _apply_flag_filter(prompt: str, flags: dict[str, bool]) -> str:
        """Keep unflagged lines; keep flagged lines only when a matching flag is set."""
        lines = []
        for line in prompt.split("\n"):
            active_flags = _FLAG_PATTERN.findall(line)
            cleaned = _FLAG_PATTERN.sub("", line).lstrip()
            if not active_flags or any(flags.get(f, False) for f in active_flags):
                lines.append(cleaned)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PromptHandler(language='{self.language}', num_prompts={len(self.data)})"
