"""Tests for configuration parsing helpers."""

from pathlib import Path

import pytest

from reme.config.config_parser import (
    _expand_env_vars,
    _load_config,
    _read_config_file,
    parse_args,
    parse_dot_notation,
    resolve_app_config,
)


def test_load_builtin_config_by_filename_with_suffix():
    """Built-in config names may include the YAML suffix."""
    cfg = _load_config("default.yaml")

    assert cfg["service"]["backend"] == "http"


def test_resolve_app_config_can_suppress_config_log(monkeypatch):
    """Client-side config resolution can avoid polluting command output."""
    messages = []

    class FakeLogger:
        """Capture config log messages."""

        def info(self, message):
            """Record one INFO message."""
            messages.append(message)

    monkeypatch.setattr("reme.utils.get_logger", lambda **_kwargs: FakeLogger())

    resolve_app_config(log_config=False)

    assert not messages


def test_default_config_registers_daily_write_job():
    """``daily_write`` is exposed as a base job backed by ``daily_write_step``."""
    cfg = _load_config("default.yaml")

    job = cfg["jobs"]["daily_write"]
    assert job["backend"] == "base"
    assert job["steps"] == [{"backend": "daily_write_step"}]
    assert job["parameters"]["required"] == ["name", "description", "session_id", "content"]


def test_default_config_keeps_frontmatter_chunk_metadata_opt_in():
    """Markdown frontmatter-to-chunk metadata is disabled by default for compatibility."""
    cfg = _load_config("default.yaml")

    markdown = cfg["components"]["file_chunker"]["markdown"]
    assert markdown["embed_toc"] is True
    assert markdown["max_ast_sections"] == 100
    assert markdown["default_chunker"] == "default"
    assert markdown["include_frontmatter_in_metadata"] is False
    # Allow-list defaults to empty; combined with the False above, chunk metadata stays empty.
    assert markdown["include_frontmatter_keys_in_metadata"] == [] or markdown.get(
        "include_frontmatter_keys_in_metadata",
    ) in (None, [])


def test_parse_args_rejects_non_key_value_extra_argument():
    """Extra CLI arguments must use key=value syntax."""
    with pytest.raises(ValueError, match="expected key=value"):
        parse_args("search", "hello")


@pytest.mark.parametrize("item", ["=1", ".a=1", "a.=1", "a..b=1"])
def test_parse_dot_notation_rejects_empty_key_segments(item):
    """Dot notation keys cannot contain empty path segments."""
    with pytest.raises(ValueError, match="Invalid dot notation key"):
        parse_dot_notation([item])


def test_read_config_file_rejects_non_mapping_root(tmp_path: Path):
    """Config files must contain a mapping at the root."""
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config root must be a mapping"):
        _read_config_file(config_path)


def test_expand_env_vars_converts_expanded_scalar_types(monkeypatch):
    """Expanded environment values keep YAML scalar typing."""
    monkeypatch.setenv("PORT", "18080")
    monkeypatch.setenv("ENABLED", "false")

    expanded = _expand_env_vars(
        {
            "port": "${PORT}",
            "enabled": "${ENABLED}",
            "zip": "${ZIP:-007}",
            "url": "http://${HOST:-localhost}:${PORT}",
            "string_bool": '${STRING_BOOL:-"false"}',
        },
    )

    assert expanded == {
        "port": 18080,
        "enabled": False,
        "zip": "007",
        "url": "http://localhost:18080",
        "string_bool": "false",
    }
