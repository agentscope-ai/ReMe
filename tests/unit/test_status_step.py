"""Tests for the built-in runtime memory status report."""

import asyncio

from reme.components.application_context import ApplicationContext
from reme.components.base_component import BaseComponent
from reme.enumeration import ComponentEnum
from reme.steps.common.status import (
    StatusStep,
    _collect_memory,
    _component_size,
)


class _SizedComponent(BaseComponent):
    """Small component with predictable owned payload for accounting tests."""

    component_type = ComponentEnum.FILE_STORE

    def __init__(self, payload: bytes, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload
        self.peer = None


def test_component_size_does_not_charge_referenced_components_twice():
    """A dependency component is accounted under its own status entry."""
    dependency = _SizedComponent(b"x" * 4096)
    owner = _SizedComponent(b"y")
    owner.peer = dependency

    owner_size = _component_size(owner)
    dependency_size = _component_size(dependency)

    assert dependency_size > owner_size


def test_collect_memory_reports_only_stateful_data_components_and_sum(tmp_path):
    """Status includes only the data components whose state can grow."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    context.components = {
        ComponentEnum.FILE_STORE: {
            "default": _SizedComponent(b"abc", app_context=context),
        },
        ComponentEnum.AGENT_WRAPPER: {
            "default": _SizedComponent(b"agent", app_context=context),
        },
        ComponentEnum.AS_LLM: {
            "default": _SizedComponent(b"llm", app_context=context),
        },
        ComponentEnum.FILE_CATALOG: {
            "default": _SizedComponent(b"catalog", app_context=context),
        },
        ComponentEnum.FILE_CHUNKER: {
            "default": _SizedComponent(b"chunker", app_context=context),
        },
        ComponentEnum.TOKENIZER: {
            "words": _SizedComponent(b"defgh", app_context=context),
        },
        ComponentEnum.FILE_GRAPH: {
            "default": _SizedComponent(b"graph", app_context=context),
        },
    }

    memory = _collect_memory(context)

    assert set(memory["components"]) == {"file_graph", "file_store"}
    assert (
        not {
            "agent_wrapper",
            "as_llm",
            "file_catalog",
            "file_chunker",
            "tokenizer",
        }
        & memory["components"].keys()
    )
    sizes = [usage["bytes"] for group in memory["components"].values() for usage in group.values()]
    assert memory["components_total_bytes"] == sum(sizes)
    assert memory["process_rss_bytes"] > 0


def test_status_step_returns_human_summary_and_exact_metadata(tmp_path):
    """The public step response serves CLI users and programmatic clients."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    context.components = {
        ComponentEnum.FILE_STORE: {
            "default": _SizedComponent(b"abc", app_context=context),
        },
    }

    response = asyncio.run(StatusStep(app_context=context)())

    assert not response.answer.startswith("ReMe status")
    assert "Memory (estimated component object size)" in response.answer
    assert "file_store:default" in response.answer
    assert "Storage" not in response.answer
    assert set(response.metadata["status"]) == {"memory"}
