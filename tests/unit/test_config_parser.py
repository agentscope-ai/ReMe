"""Tests for configuration parsing helpers."""

from pathlib import Path

import pytest

from reme.config.config_parser import (
    _expand_env_vars,
    _load_config,
    _read_config_file,
    parse_args,
    parse_dot_notation,
)


def test_load_builtin_config_by_filename_with_suffix():
    """Built-in config names may include the YAML suffix."""
    cfg = _load_config("default.yaml")

    assert cfg["service"]["backend"] == "http"


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
