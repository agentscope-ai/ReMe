"""Tests for installed plugin discovery and application-local registration."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

from pathlib import Path

import pytest

from reme.application import Application
from reme.components.base_component import BaseComponent, ComponentMixin
from reme.components.component_registry import ComponentRegistry, R
from reme.config.config_parser import _load_config
from reme.enumeration import ComponentEnum
from reme.plugin import Backend, Plugin, PluginManager, _load_backend
from reme.plugin_manifest import parse_plugin_manifest


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


@pytest.mark.parametrize("name", ["lme", "beam"])
def test_shared_benchmark_preset_is_builtin_but_plugin_aliases_and_backends_are_not(monkeypatch, name):
    _set_entry_points(monkeypatch)
    benchmark = _load_config("benchmark")
    assert {"index_update", "digest_update", "read", "write"} <= benchmark["jobs"].keys()
    assert {"search", "auto_memory", "agentic_answer", "answer_judge"}.isdisjoint(benchmark["jobs"])
    for alias in (name, f"{name}.yaml"):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load_config(alias)
    assert R.get(ComponentEnum.STEP, f"{name}_auto_memory_step") is None
    assert R.get(ComponentEnum.STEP, f"{name}_agentic_answer_step") is None
    assert R.get(ComponentEnum.STEP, f"{name}_search_v2_step") is None
    judge = "lme_answer_judge_step" if name == "lme" else "beam_rubric_judge_step"
    assert R.get(ComponentEnum.STEP, judge) is None


def test_plugin_application_defaults_are_below_application_config():
    manager = PluginManager(
        [
            Plugin(
                name="example",
                config={
                    "language": "plugin",
                    "jobs": {"task": {"backend": "plugin_job", "plugin_only": True}},
                },
            ),
        ],
    )

    merged = manager.merge_config(
        {
            "language": "application",
            "jobs": {"task": {"backend": "application_job", "application_only": True}},
        },
    )

    assert merged["language"] == "application"
    assert merged["jobs"]["task"] == {"backend": "plugin_job", "plugin_only": True}


def test_later_plugin_atomically_replaces_same_name_job_and_records_each_collision():
    manager = PluginManager(
        [
            Plugin(name="first", config={"jobs": {"task": {"backend": "first", "first_only": True}}}),
            Plugin(name="second", config={"jobs": {"task": {"backend": "second", "second_only": True}}}),
        ],
    )

    merged, warnings = manager._merge_config(  # pylint: disable=protected-access
        {"jobs": {"task": {"backend": "application", "application_only": True}}},
    )

    assert merged["jobs"]["task"] == {"backend": "second", "second_only": True}
    assert warnings == (
        "Config collision at jobs.task: plugin 'first' replaces the complete definition from application config",
        "Config collision at jobs.task: plugin 'second' replaces the complete definition from plugin 'first'",
    )


def test_identical_same_name_job_warns_while_different_names_are_merged():
    definition = {"backend": "same"}
    manager = PluginManager(
        [Plugin(name="example", config={"jobs": {"shared": definition, "plugin_only": definition}})],
    )

    merged, warnings = manager._merge_config(  # pylint: disable=protected-access
        {"jobs": {"shared": definition, "application_only": definition}},
    )

    assert set(merged["jobs"]) == {"shared", "plugin_only", "application_only"}
    expected = "Config collision at jobs.shared: " + (
        "plugin 'example' replaces the complete definition from application config"
    )
    assert warnings == (expected,)


def test_plugin_component_identity_is_normalized_before_atomic_replacement():
    manager = PluginManager(
        [
            Plugin(
                name="example",
                config={
                    "components": {
                        " as_llm ": {
                            "default": {"backend": "plugin", "plugin_only": True},
                        },
                    },
                },
            ),
        ],
    )

    merged, warnings = manager._merge_config(  # pylint: disable=protected-access
        {
            "components": {
                "as_llm": {
                    "default": {"backend": "application", "application_only": True},
                    "other": {"backend": "other"},
                },
            },
        },
    )

    assert merged["components"]["as_llm"] == {
        "default": {"backend": "plugin", "plugin_only": True},
        "other": {"backend": "other"},
    }
    assert warnings == (
        "Config collision at components.as_llm.default: "
        "plugin 'example' replaces the complete definition from application config",
    )


def test_later_plugin_atomically_replaces_same_name_component():
    manager = PluginManager(
        [
            Plugin(
                name="first",
                config={"components": {"tokenizer": {"default": {"backend": "first", "first_only": True}}}},
            ),
            Plugin(
                name="second",
                config={"components": {"tokenizer": {"default": {"backend": "second"}}}},
            ),
        ],
    )

    merged, warnings = manager._merge_config({})  # pylint: disable=protected-access

    assert merged["components"]["tokenizer"]["default"] == {"backend": "second"}
    assert warnings == (
        "Config collision at components.tokenizer.default: "
        "plugin 'second' replaces the complete definition from plugin 'first'",
    )


def test_same_component_name_in_different_types_does_not_conflict():
    manager = PluginManager(
        [Plugin(name="example", config={"components": {"tokenizer": {"default": {"backend": "plugin"}}}})],
    )

    merged, warnings = manager._merge_config(  # pylint: disable=protected-access
        {"components": {"as_llm": {"default": {"backend": "application"}}}},
    )

    assert merged["components"] == {
        "as_llm": {"default": {"backend": "application"}},
        "tokenizer": {"default": {"backend": "plugin"}},
    }
    assert not warnings


def test_same_source_component_type_normalization_uses_later_definition_without_warning():
    manager = PluginManager(
        [
            Plugin(
                name="example",
                config={
                    "components": {
                        "as_llm ": {"default": {"backend": "first"}},
                        "as_llm": {"default": {"backend": "second"}},
                    },
                },
            ),
        ],
    )

    merged, warnings = manager._merge_config({})  # pylint: disable=protected-access

    assert merged["components"] == {"as_llm": {"default": {"backend": "second"}}}
    assert not warnings


def test_plugin_application_defaults_expand_environment(monkeypatch):
    monkeypatch.setenv("PLUGIN_LIMIT", "12")
    manager = PluginManager(
        [
            Plugin(
                name="example",
                config={
                    "limit": "${PLUGIN_LIMIT}",
                    "jobs": {"task": {"backend": "base", "limit": "${PLUGIN_LIMIT}"}},
                },
            ),
        ],
    )

    merged = manager.merge_config({})

    assert merged["limit"] == 12
    assert merged["jobs"]["task"]["limit"] == 12


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


def test_application_logs_config_collisions_before_resource_instantiation(monkeypatch, tmp_path):
    events = []
    manager = PluginManager(
        [Plugin(name="example", config={"jobs": {"task": {"backend": "plugin"}}})],
    )

    class RecordingLogger:
        def bind(self, **_kwargs):
            return self

        def warning(self, message):
            events.append(("warning", message))

        def info(self, _message):
            events.append(("info", None))

    monkeypatch.setattr(PluginManager, "discover", classmethod(lambda cls, specs: manager))
    monkeypatch.setattr("reme.application.get_logger", lambda **_kwargs: RecordingLogger())
    monkeypatch.setattr("reme.components.base_component.get_logger", lambda **_kwargs: RecordingLogger())
    monkeypatch.setattr(Application, "_init_service", lambda self: events.append(("service", None)))
    monkeypatch.setattr(Application, "_init_components", lambda self: events.append(("components", None)))
    monkeypatch.setattr(Application, "_init_jobs", lambda self: events.append(("jobs", None)))

    Application(
        plugins=["example"],
        jobs={"task": {"backend": "application"}},
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "unused"},
    )

    assert events[0] == (
        "warning",
        "Config collision at jobs.task: plugin 'example' replaces the complete definition from application config",
    )
    assert [kind for kind, _ in events if kind in {"service", "components", "jobs"}] == [
        "service",
        "components",
        "jobs",
    ]


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


def test_plugin_manager_loads_package_manifest(monkeypatch, tmp_path):
    package = tmp_path / "example_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "backend.py").write_text(
        "from reme.components.base_component import ComponentMixin\n"
        "from reme.enumeration import ComponentEnum\n"
        "class ExampleStep(ComponentMixin):\n"
        "    component_type = ComponentEnum.STEP\n",
        encoding="utf-8",
    )
    (package / "plugin.yaml").write_text(
        "backends:\n"
        "  example_step: example_plugin.backend:ExampleStep\n"
        "application_defaults:\n"
        "  jobs:\n"
        "    example:\n"
        "      backend: base\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _set_entry_points(
        monkeypatch,
        _FakeEntryPoint("example", "example_plugin", lambda: None, "reme.plugins"),
    )

    manager = PluginManager.discover(["example"])
    registry = ComponentRegistry()
    manager.register(registry)

    backend = registry.get(ComponentEnum.STEP, "example_step")
    assert backend is not None
    assert backend.__name__ == "ExampleStep"
    assert manager.merge_config({})["jobs"]["example"]["backend"] == "base"


def test_plugin_manifest_rejects_legacy_defaults_field():
    with pytest.raises(ValueError, match="unknown keys: defaults"):
        parse_plugin_manifest("defaults: {}\n", plugin_name="example")


def test_plugin_manifest_requires_application_defaults_mapping():
    with pytest.raises(TypeError, match="manifest 'application_defaults' must be a mapping"):
        parse_plugin_manifest("application_defaults: []\n", plugin_name="example")


def test_plugin_manifest_reports_missing_backend_attribute():
    with pytest.raises(ValueError, match="cannot load backend.*MissingStep"):
        _load_backend("reme.plugin:MissingStep", plugin_name="missing")


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
