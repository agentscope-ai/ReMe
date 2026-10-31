"""Real Application/Job/FileStore wiring with deterministic local model boundaries."""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
# pylint: disable=unused-argument,redefined-outer-name

from importlib.metadata import EntryPoint

import numpy as np
import pytest
import pytest_asyncio

from reme.application import Application
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.embedding_store import BaseEmbeddingStore
from reme.plugin import Backend, Plugin, PluginManager, _load_plugin


class FakeEmbedding(BaseEmbeddingStore):
    dimensions = 3
    vector_space_id = "procedure-test"

    async def health_check(self, timeout=None):
        return True

    async def get_embeddings(self, input_text, **kwargs):
        return [
            np.array([1, text.count("repair") + 1, text.count("index") + 1], dtype=np.float16) for text in input_text
        ]


class FakeAgent(BaseAgentWrapper):
    async def reply(self, inputs, **kwargs):
        return await self.app_context.metadata["fake_reply"](inputs, kwargs)


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1")
    return _load_plugin(
        "procedure-memory",
        EntryPoint(name="procedure-memory", value="reme_procedure_memory", group="reme.plugins"),
    )


@pytest_asyncio.fixture
async def app_factory(monkeypatch, tmp_path, plugin):
    manager = PluginManager(
        (
            plugin,
            Plugin(name="test", backends=(Backend("fake_embedding", FakeEmbedding), Backend("fake_agent", FakeAgent))),
        ),
    )
    monkeypatch.setattr(PluginManager, "discover", classmethod(lambda cls, specs: manager))
    apps = []

    async def create(workspace=None, **store_options):
        app = Application(
            workspace_dir=str(workspace or tmp_path / "workspace"),
            plugins=["procedure-memory"],
            service={"backend": "cli"},
            enable_logo=False,
            log_to_console=False,
            log_to_file=False,
            components={
                "embedding_store": {"procedure": {"backend": "fake_embedding"}},
                "agent_wrapper": {"procedure": {"backend": "fake_agent"}},
                "procedure_memory": {"default": {**store_options}},
            },
        )
        store = app.context.components["procedure_memory"]["default"]
        store.budget.counter = len  # Exact deterministic character tokenizer; never download model files.
        await app.start()
        apps.append(app)
        return app, store

    yield create
    for app in reversed(apps):
        await app.close()


@pytest_asyncio.fixture
async def runtime(app_factory, tmp_path):
    app, store = await app_factory()
    source = tmp_path / "input.log"
    source.write_text("repair an index\n", encoding="utf-8")
    imported = store.import_trajectory(str(source), "input.log", "fixture")
    scope = store.open_scope(imported["source"], 0)
    return app, store, scope


@pytest.fixture
def experience():
    return {
        "filename": "repair-search-index.json",
        "description": "Repair search",
        "when-to-use": "After an index configuration change",
        "content": "Rebuild the index before searching",
        "operation_key": "create-repair",
    }
