"""Plugin manifest, schema and lifecycle integration."""

# pylint: disable=missing-function-docstring,protected-access

import json
import tomllib
from pathlib import Path

import pytest

from reme_procedure_memory.schema import ProcedureMemory

from reme.components.component_registry import R
from reme.enumeration import ComponentEnum
from reme.plugin import PluginManager


def test_manifest_and_package(plugin):
    package = Path(__file__).parents[1]
    config = tomllib.loads((package / "pyproject.toml").read_text())
    assert config["project"]["entry-points"]["reme.plugins"] == {"procedure-memory": "reme_procedure_memory"}
    assert len(plugin.backends) == 7
    assert R.get(ComponentEnum.STEP, "procedure_auto_memory_step") is None
    merged = PluginManager([plugin]).merge_config(
        {"components": {"procedure_memory": {"default": {"procedure_dir": "custom"}}}},
    )
    assert merged["components"]["procedure_memory"]["default"]["procedure_dir"] == "custom"


def test_four_field_contract():
    document = {"description": "d", "when-to-use": "w", "content": "c", "sources": ["session/trajectories/a.log"]}
    assert ProcedureMemory.model_validate(document).document() == document
    with pytest.raises(ValueError):
        ProcedureMemory.model_validate({**document, "experience_id": "no"})


@pytest.mark.asyncio
async def test_lifecycle_tools_and_json_answers(runtime, experience):
    app, store, scope = runtime
    assert store.is_started
    assert not app.context.jobs["procedure_auto_memory"].enable_serve
    assert not app.context.jobs["procedure_create"].enable_serve
    assert app.context.jobs["procedure_search"].enable_serve
    rejected = await app.context.jobs["procedure_create"](**experience)
    assert not rejected.success
    created = await app.context.jobs["procedure_create"](**experience, _procedure_scope=scope)
    assert created.success
    result = await app.context.jobs["procedure_read"](path=created.metadata["path"])
    assert json.loads(result.answer) == result.metadata
    assert set(result.metadata["experience"]) == {"description", "when-to-use", "content", "sources"}


@pytest.mark.asyncio
async def test_configured_directories(app_factory, tmp_path):
    _, store = await app_factory(procedure_dir="knowledge/procedures", trajectory_subdir="raw")
    source = tmp_path / "a.txt"
    source.write_text("a")
    imported = store.import_trajectory(str(source), "a.txt", "custom")
    assert imported["source"] == "session/raw/a.txt"
    assert store.procedure_root.name == "procedures"
