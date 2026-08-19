"""Tests for installed plugin discovery and application-local registration."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from pathlib import Path

import pytest

from reme.application import Application
from reme.components.base_component import BaseComponent, ComponentMixin
from reme.components.component_registry import ComponentRegistry, R
from reme.config.config_parser import _load_config
from reme.enumeration import ComponentEnum
from reme.plugin import Backend, Plugin, PluginManager


class _PluginStep(ComponentMixin):
    component_type = ComponentEnum.STEP


class _PluginComponent(BaseComponent):
    component_type = "example.reranker"


class _FakeEntryPoint:
    def __init__(self, name, value, loader, group):
        self.name = name
        self.value = value
        self._loader = loader
        self.group = group

    def load(self):
        return self._loader()


class _FakeEntryPoints(list):
    def select(self, *, group, name):
        return [entry for entry in self if entry.group == group and entry.name == name]


def _set_entry_points(monkeypatch, *entries):
    monkeypatch.setattr("reme.entry_point.metadata.entry_points", lambda: _FakeEntryPoints(entries))


def test_plugin_defaults_are_below_application_config():
    manager = PluginManager(
        [
            Plugin(
                name="example",
                config={"jobs": {"task": {"backend": "base", "value": 1}}},
            ),
        ],
    )

    merged = manager.merge_config({"jobs": {"task": {"value": 2}}})

    assert merged["jobs"]["task"] == {"backend": "base", "value": 2}


def test_plugin_defaults_expand_environment(monkeypatch):
    monkeypatch.setenv("PLUGIN_LIMIT", "12")
    manager = PluginManager([Plugin(name="example", config={"limit": "${PLUGIN_LIMIT}"})])

    assert manager.merge_config({})["limit"] == 12


def test_plugin_registers_into_only_the_supplied_registry():
    manager = PluginManager([Plugin(name="example", backends=(Backend("example_step", _PluginStep),))])
    first = ComponentRegistry()
    second = ComponentRegistry()

    manager.register(first)

    assert first.get(ComponentEnum.STEP, "example_step") is _PluginStep
    assert second.get(ComponentEnum.STEP, "example_step") is None


@pytest.mark.asyncio
async def test_plugin_registers_and_runs_custom_component_type(monkeypatch, tmp_path):
    manager = PluginManager(
        [
            Plugin(
                name="example",
                backends=(Backend("cross_encoder", _PluginComponent),),
                config={
                    "components": {
                        "example.reranker": {
                            "default": {"backend": "cross_encoder"},
                        },
                    },
                },
            ),
        ],
    )
    monkeypatch.setattr(PluginManager, "discover", classmethod(lambda cls, specs: manager))

    app = Application(
        plugins=["example"],
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
    )

    component = app.context.components["example.reranker"]["default"]
    assert isinstance(component, _PluginComponent)
    assert app.context.registry.get("example.reranker", "cross_encoder") is _PluginComponent

    await app.start()
    assert component.is_started is True
    await app.update_component("example.reranker", "default", backend="updated")
    assert component.backend == "updated"
    await app.close()
    assert component.is_started is False


def test_plugin_backend_collision_fails_with_both_owners():
    registry = ComponentRegistry()
    registry.add("same", _PluginStep, owner="first")

    class OtherStep(ComponentMixin):
        component_type = ComponentEnum.STEP

    with pytest.raises(ValueError, match="both 'first' and 'second'"):
        registry.add("same", OtherStep, owner="second")


def test_plugin_manager_loads_explicit_entry_point(monkeypatch):
    descriptor = Plugin(name="example", backends=(Backend("example_step", _PluginStep),))
    _set_entry_points(
        monkeypatch,
        _FakeEntryPoint("example", "example:plugin", lambda: descriptor, "reme.plugins"),
    )

    manager = PluginManager.discover(["example"])

    assert manager.plugins == (descriptor,)


def test_plugin_manager_rejects_multiple_entry_point_providers(monkeypatch):
    descriptor = Plugin(name="example")
    _set_entry_points(
        monkeypatch,
        _FakeEntryPoint("example", "first:plugin", lambda: descriptor, "reme.plugins"),
        _FakeEntryPoint("example", "second:plugin", lambda: descriptor, "reme.plugins"),
    )

    with pytest.raises(
        ValueError,
        match="Plugin 'example' has multiple installed providers: first:plugin, second:plugin",
    ):
        PluginManager.discover(["example"])


def test_plugin_entry_point_import_side_effect_does_not_leak(monkeypatch, tmp_path):
    class UndeclaredClient(ComponentMixin):
        component_type = ComponentEnum.CLIENT

    descriptor = Plugin(name="example")

    def load_plugin():
        R.register(UndeclaredClient, "undeclared-client")
        return descriptor

    _set_entry_points(
        monkeypatch,
        _FakeEntryPoint("example", "example:plugin", load_plugin, "reme.plugins"),
    )

    app = Application(
        plugins=["example"],
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
    )

    assert R.get(ComponentEnum.CLIENT, "undeclared-client") is None
    assert app.context.registry.get(ComponentEnum.CLIENT, "undeclared-client") is None


def test_plugin_manager_rejects_non_string_name():
    with pytest.raises(TypeError, match="Invalid plugin name"):
        PluginManager.discover([{"name": "example"}])


def test_config_can_extend_another_config(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("service:\n  backend: http\n  port: 8000\n", encoding="utf-8")
    child.write_text("extends: parent.yaml\nservice:\n  port: 9000\n", encoding="utf-8")

    assert _load_config(str(child))["service"] == {"backend": "http", "port": 9000}


def test_config_can_come_from_installed_entry_point(tmp_path: Path, monkeypatch):
    config = tmp_path / "example.yaml"
    config.write_text("plugins: [example]\n", encoding="utf-8")
    _set_entry_points(
        monkeypatch,
        _FakeEntryPoint("example", "example:CONFIG_PATH", lambda: config, "reme.configs"),
    )

    assert _load_config("example") == {"plugins": ["example"]}


def test_config_entry_point_import_side_effect_does_not_leak(tmp_path: Path, monkeypatch):
    config = tmp_path / "side-effect.yaml"
    config.write_text("service:\n  backend: cli\n", encoding="utf-8")

    class UndeclaredClient(ComponentMixin):
        component_type = ComponentEnum.CLIENT

    def load_config():
        R.register(UndeclaredClient, "config-side-effect-client")
        return config

    _set_entry_points(
        monkeypatch,
        _FakeEntryPoint("side-effect", "example:CONFIG_PATH", load_config, "reme.configs"),
    )

    loaded = _load_config("side-effect")
    app = Application(
        **loaded,
        workspace_dir=str(tmp_path / "workspace"),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )

    assert loaded == {"service": {"backend": "cli"}}
    assert R.get(ComponentEnum.CLIENT, "config-side-effect-client") is None
    assert app.context.registry.get(ComponentEnum.CLIENT, "config-side-effect-client") is None
