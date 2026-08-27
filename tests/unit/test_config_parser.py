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
    assert cfg["service"]["mcp_enabled"] is True
    assert cfg["service"]["mcp_path"] == "/mcp"


@pytest.mark.parametrize("provider_count", [1, 2])
def test_builtin_and_external_config_name_collision_fails(monkeypatch, provider_count):
    """An installed config cannot be silently shadowed by a built-in name."""

    class FakeEntryPoint:
        """Installed config entry point with a built-in name."""

        name = "default"
        value = "example:CONFIG_PATH"

        @staticmethod
        def load():
            """The provider need not be imported to detect the collision."""
            raise AssertionError("colliding provider should not be loaded")

    class FakeEntryPoints(list):
        """Minimal selectable entry-point collection."""

        def select(self, *, group, name):
            """Return entries matching the requested group and name."""
            assert group == "reme.configs"
            return [entry for entry in self if entry.name == name]

    monkeypatch.setattr(
        "reme.entry_point.metadata.entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint() for _ in range(provider_count)]),
    )

    with pytest.raises(ValueError, match="provided by both ReMe and an installed distribution"):
        _load_config("default")


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


def test_resolve_app_config_layers_plugins_over_default():
    """A plugin-only start keeps the ordinary default application config."""
    config = resolve_app_config(log_config=False, plugins=["auto-fin"])

    assert config["service"]["backend"] == "http"
    assert config["plugins"] == ["auto-fin"]


def test_default_config_registers_daily_write_job():
    """``daily_write`` is exposed as a base job backed by ``daily_write_step``."""
    cfg = _load_config("default.yaml")

    job = cfg["jobs"]["daily_write"]
    assert job["backend"] == "base"
    assert job["steps"] == [{"backend": "daily_write_step"}]
    assert job["parameters"]["required"] == ["name", "description", "session_id", "content"]


def test_default_config_registers_app_config_job():
    """The default service exposes the effective application config."""
    job = _load_config("default.yaml")["jobs"]["app_config"]

    assert job["backend"] == "base"
    assert job["steps"] == [{"backend": "app_config_step"}]


def test_default_config_registers_workspace_web_jobs():
    """The local web client has stable save and streaming chat actions."""
    jobs = _load_config("default.yaml")["jobs"]

    assert jobs["save"]["steps"] == [{"backend": "save_step"}]
    assert jobs["load"]["steps"] == [{"backend": "load_step", "max_bytes": 5242880}]
    assert jobs["chat"]["backend"] == "stream"
    assert jobs["chat"]["steps"] == [{"backend": "chat_step", "agent_wrapper": "default"}]


def test_default_config_keeps_frontmatter_chunk_metadata_opt_in():
    """Markdown frontmatter-to-chunk metadata is disabled by default for compatibility."""
    cfg = _load_config("default.yaml")

    markdown = cfg["components"]["file_chunker"]["markdown"]
    assert markdown["embed_toc"] is True
    assert markdown["max_ast_sections"] == 100
    assert markdown["include_frontmatter_in_metadata"] is False
    # Allow-list defaults to empty; combined with the False above, chunk metadata stays empty.
    assert markdown["include_frontmatter_keys_in_metadata"] == [] or markdown.get(
        "include_frontmatter_keys_in_metadata",
    ) in (None, [])


def test_daily_cookbook_chunks_jsonl_one_line_at_a_time():
    """Daily cookbook keeps JSONL records as individually addressable chunks."""
    cfg = _load_config("daily_cookbook.yaml")

    jsonl = cfg["components"]["file_chunker"]["jsonl"]
    assert jsonl["max_lines_per_chunk"] == 1


def test_parse_args_rejects_non_key_value_extra_argument():
    """Extra CLI arguments must use key=value syntax."""
    with pytest.raises(ValueError, match="expected key=value"):
        parse_args("search", "hello")


def test_parse_args_separates_action_and_application_kwargs():
    """The shared action grammar is independent from application key/value parsing."""
    action, kwargs = parse_args("--search", "--query=hello", "limit=3")

    assert action == "search"
    assert kwargs == {"query": "hello", "limit": 3}


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
