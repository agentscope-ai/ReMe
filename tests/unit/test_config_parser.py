"""Tests for configuration parsing helpers."""

from pathlib import Path

import pytest

from reme.config.config_parser import (
    expand_env_vars,
    _load_config,
    _read_config_file,
    parse_args,
    parse_dot_notation,
    parse_kwargs,
    resolve_app_config,
)


def test_load_builtin_config_by_filename_with_suffix():
    """Built-in config names may include the YAML suffix."""
    assert _load_config("default.yaml") == _load_config("default")


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
        "reme.core.entry_point.metadata.entry_points",
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

    monkeypatch.setattr("reme.utils.logger_utils.get_logger", lambda **_kwargs: FakeLogger())

    resolve_app_config(log_config=False)

    assert not messages


def test_resolve_app_config_layers_plugins_over_default():
    """A plugin-only start keeps the ordinary default application config."""
    config = resolve_app_config(log_config=False, plugins=["auto-fin"])

    assert config == {**_load_config("default"), "plugins": ["auto-fin"]}


def test_yaml_aliases_keep_job_overrides_isolated(tmp_path):
    """YAML aliases share definitions without coupling loaded job data or CLI overrides."""
    path = tmp_path / "jobs.yaml"
    path.write_text(
        "jobs:\n  manual:\n    steps: &steps [{backend: read}]\n  scheduled:\n    steps: *steps\n",
        encoding="utf-8",
    )
    jobs = _load_config(str(path))["jobs"]
    jobs["manual"]["steps"][0]["backend"] = "edit"
    assert jobs["scheduled"]["steps"] == [{"backend": "read"}]

    overrides = parse_kwargs(f"config={path}", 'jobs.manual.steps=[{"backend":"write"}]')
    resolved = resolve_app_config(log_config=False, **overrides)["jobs"]
    assert resolved["manual"]["steps"] == [{"backend": "write"}]
    assert resolved["scheduled"]["steps"] == jobs["scheduled"]["steps"]


def test_parse_args_rejects_non_key_value_extra_argument():
    """Extra CLI arguments must use key=value syntax."""
    with pytest.raises(ValueError, match="expected key=value"):
        parse_args("search", "hello")


def test_parse_args_separates_action_and_application_kwargs():
    """The shared action grammar is independent from application key/value parsing."""
    action, kwargs = parse_args("--search", "--query=hello", "limit=3")

    assert action == "search"
    assert kwargs == {"query": "hello", "limit": 3}


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-Infinity", "1e999"])
def test_parse_kwargs_keeps_non_finite_numbers_as_strings(value):
    """JSON has no NaN or Infinity, so these values stay text instead of becoming floats."""
    assert parse_kwargs(f"query={value}", "limit=5", "min_score=0.5") == {
        "query": value,
        "limit": 5,
        "min_score": 0.5,
    }


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

    expanded = expand_env_vars(
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
