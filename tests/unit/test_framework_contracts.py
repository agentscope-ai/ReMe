"""Public import, backend, CLI and plugin contracts retained during framework refactoring."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

import reme
from reme.components import ComponentMixin, R, create_application_registry
from reme.entry_point import CONFIG_ENTRY_POINT_GROUP, PLUGIN_ENTRY_POINT_GROUP

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = json.loads((Path(__file__).parent / "fixtures" / "builtin_backends.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("public_module", "definition_module", "names"),
    [
        ("reme", "reme.application", ("Application",)),
        ("reme", "reme.reme", ("ReMe",)),
        ("reme", "reme.components", ("BaseComponent",)),
        ("reme", "reme.plugin", ("Backend", "Plugin")),
        ("reme.components", "reme.components.application_context", ("ApplicationContext",)),
        ("reme.components", "reme.components.runtime_context", ("RuntimeContext",)),
        ("reme.components", "reme.components.base_component", ("BaseComponent", "ComponentMixin")),
        (
            "reme.components",
            "reme.components.component_registry",
            ("ComponentRegistry", "R", "create_application_registry"),
        ),
        ("reme.components", "reme.components.prompt_handler", ("PromptHandler",)),
        ("reme.components.client", "reme.components.client.base_client", ("BaseClient",)),
        ("reme.components.client", "reme.components.client.http_client", ("HttpClient",)),
        ("reme.components.client", "reme.components.client.mcp_client", ("MCPClient",)),
        ("reme.steps", "reme.steps.base_step", ("BaseStep",)),
        ("reme.schema", "reme.schema.application_config", ("ApplicationConfig", "ComponentConfig", "JobConfig")),
        ("reme.schema", "reme.schema.request", ("Request",)),
        ("reme.schema", "reme.schema.response", ("Response",)),
        ("reme.schema", "reme.schema.stream_chunk", ("StreamChunk",)),
        ("reme.enumeration", "reme.enumeration.component_enum", ("ComponentEnum",)),
        ("reme.enumeration", "reme.enumeration.chunk_enum", ("ChunkEnum",)),
    ],
)
def test_public_exports_preserve_object_identity(public_module, definition_module, names):
    """Compatibility imports must refer to one definition rather than duplicate classes or state."""
    public = importlib.import_module(public_module)
    definition = importlib.import_module(definition_module)
    for name in names:
        assert name in public.__all__
        assert getattr(public, name) is getattr(definition, name), f"{public_module}.{name} lost its identity"


@pytest.mark.parametrize(("component_type", "names"), BACKENDS.items())
def test_builtin_backend_names_remain_available(component_type, names):
    """Existing configuration names survive moves, without pinning implementation module paths."""
    application_registry = create_application_registry()
    for name in names:
        implementation = R.get(component_type, name)
        assert implementation is not None, f"Missing built-in backend: {component_type}:{name}"
        assert issubclass(implementation, ComponentMixin)
        assert implementation.component_type == component_type
        assert application_registry.get(component_type, name) is implementation


def test_console_entrypoint_remains_resolvable():
    """Existing installed launchers keep their configured import target."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"]["reme"] == "reme.reme:main"
    module_name, name = project["scripts"]["reme"].split(":")
    assert callable(getattr(importlib.import_module(module_name), name))


def test_plugin_entrypoint_groups_remain_stable():
    """Published plugin packages keep both contribution discovery names."""
    assert PLUGIN_ENTRY_POINT_GROUP == "reme.plugins"
    assert CONFIG_ENTRY_POINT_GROUP == "reme.configs"


def test_applications_isolate_registries_and_application_state(tmp_path):
    """An embedded application cannot mutate another application's providers or metadata."""
    options = {
        "service": {"backend": "cli"},
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
    }
    first = reme.Application(workspace_dir=str(tmp_path / "first"), **options)
    second = reme.Application(workspace_dir=str(tmp_path / "second"), **options)
    first.context.metadata["baseline.calls"] = ["first"]
    assert "baseline.calls" not in second.context.metadata
    assert first.context.registry is not second.context.registry
    assert first.context.registry.unregister("step", "version_step")
    assert second.context.registry.get("step", "version_step") is R.get("step", "version_step")
    assert R.get("step", "version_step") is not None


def test_module_entrypoint_runs_one_shot_job_in_an_isolated_workspace(tmp_path):
    """Exercise module startup, config loading and a real Job without a server or model."""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "workspace_dir": str(tmp_path / "workspace"),
                "service": {"backend": "http"},
                "enable_logo": False,
                "log_to_console": False,
                "log_to_file": False,
                "jobs": {"version": {"backend": "base", "steps": [{"backend": "version_step"}]}},
            },
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), environment.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-m", "reme.reme", "start", f"config={config}", "job=version"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == reme.__version__
    assert (tmp_path / "workspace").is_dir()
