"""Compatibility tests for applications that embed ReMe in-process."""

# pylint: disable=protected-access

import asyncio
from unittest.mock import Mock, patch

import pytest

from reme import ReMe
from reme.components.agent_wrapper import AsAgentWrapper
from reme.components.as_llm import BaseAsLLM, DashScopeAsLLM
from reme.enumeration import ComponentEnum
from reme.schema import FileNode


def _qwenpaw_style_config(workspace_dir: str) -> dict:
    """Return the narrow ReMe contract used by QwenPaw's memory manager."""
    return {
        "workspace_dir": workspace_dir,
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
        "service": {"backend": "http"},
        "jobs": {
            "version": {
                "backend": "base",
                "description": "return reme package version",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "version_step"}],
            },
        },
        "components": {
            "as_llm": {
                "default": {
                    "backend": "openai",
                    "model": "consumer-injected",
                    "credential": {"api_key": "", "base_url": ""},
                },
            },
            "agent_wrapper": {
                "default": {
                    "backend": "agentscope",
                    "as_llm": "default",
                },
            },
        },
    }


def _file_graph_config(workspace_dir: str) -> dict:
    """Return a minimal application with one persistent file graph."""
    return {
        "workspace_dir": workspace_dir,
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
        "service": {"backend": "http"},
        "components": {"file_graph": {"default": {"backend": "local"}}},
    }


def test_qwenpaw_style_config_preserves_optional_defaults(tmp_path):
    """New application fields remain optional for existing embedded configs."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))

    assert app.config.environment == {}
    assert app.context.service is not None
    assert app.context.service.jobs is None

    wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]
    assert isinstance(wrapper, AsAgentWrapper)
    assert wrapper.subprocess_environment == {}


def test_qwenpaw_style_config_keeps_in_process_application_api(tmp_path):
    """Model injection, lifecycle, and direct job execution remain compatible."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    injected_model = object()

    async def exercise_api() -> None:
        component = await app.update_component(
            "as_llm",
            "default",
            model=injected_model,
        )
        await app.start()
        try:
            response = await app.run_job("version")

            assert component.model is injected_model
            assert response.success is True
            assert response.answer
        finally:
            await app.close()

        assert app.is_started is False

    asyncio.run(exercise_api())


@pytest.mark.parametrize("finish_by_timeout", [False, True], ids=["graceful", "job-timeout"])
def test_cancelled_embedded_close_waits_for_worker_before_store(tmp_path, monkeypatch, finish_by_timeout):
    """A host cancelling close must not tear down a live indexing worker's store."""
    config = {
        "workspace_dir": str(tmp_path),
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
        "thread_pool_max_workers": 1,
        "service": {"backend": "cli"},
        "components": {
            "file_store": {"default": {"backend": "local", "embedding_store": ""}},
            "file_graph": {"default": {"backend": "local"}},
            "keyword_index": {"default": {"backend": "bm25"}},
            "tokenizer": {"default": {"backend": "regex"}},
        },
        "jobs": {"index": {"backend": "background", "supervisor": False, "close_timeout": 5}},
    }

    async def exercise_api():
        app = ReMe(**config)
        job = app.context.jobs["index"]
        store = app.context.components[ComponentEnum.FILE_STORE]["default"]
        running = asyncio.Event()
        stopping = asyncio.Event()
        release = asyncio.Event()
        worker_stopped = asyncio.Event()
        store_close_observations = []

        async def pending_index_update(context):
            running.set()
            try:
                await context.stop_event.wait()
                stopping.set()
                await release.wait()
                await store.upsert([(FileNode(path="memory.md", st_mtime=1.0), [])])
            finally:
                worker_stopped.set()

        original_store_close = store._close

        async def observe_store_close():
            store_close_observations.append(worker_stopped.is_set())
            await original_store_close()

        monkeypatch.setattr(job, "_build_steps", lambda: [pending_index_update])
        monkeypatch.setattr(store, "_close", observe_store_close)
        await app.start()
        worker = job._task
        pool = app.context.thread_pool
        close_task = None
        try:
            await asyncio.wait_for(running.wait(), timeout=5)
            if finish_by_timeout:
                job.close_timeout = 0.2
            close_task = asyncio.create_task(app.close())
            await asyncio.wait_for(stopping.wait(), timeout=5)
            # QwenPaw cancels the closing task when its shared deadline expires.
            for attempt in range(2):
                close_task.cancel(f"host deadline {attempt}")
                await asyncio.sleep(0)
            assert not close_task.done()
            assert not store_close_observations
            assert store.is_started
            assert not worker.done()
            if not finish_by_timeout:
                release.set()
            with pytest.raises(asyncio.CancelledError, match="host deadline 0"):
                await asyncio.wait_for(close_task, timeout=5)

            assert worker.done()
            assert worker.cancelled() is finish_by_timeout
            assert job._task is None
            assert store_close_observations == [True]
            assert not app._started_components
            assert not app.is_started
            assert app.context.thread_pool is None
            assert all(not c.is_started for group in app.context.components.values() for c in group.values())
            if not finish_by_timeout:
                config["jobs"] = {}
                async with ReMe(**config) as restored:
                    restored_store = restored.context.components[ComponentEnum.FILE_STORE]["default"]
                    assert [node.path for node in await restored_store.get_nodes()] == ["memory.md"]
        finally:
            release.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            if close_task is not None:
                close_task.cancel()
                await asyncio.gather(close_task, return_exceptions=True)
            await app._close_started_components()
            pool.shutdown(wait=True)

    asyncio.run(exercise_api())


@pytest.mark.parametrize("injected", [False, True], ids=["uninitialized", "injected"])
def test_update_component_validates_all_fields_before_mutation(tmp_path, injected):
    """Rejected updates preserve model state without constructing a provider."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    component = app.context.components[ComponentEnum.AS_LLM]["default"]
    original_model = object() if injected else None
    if injected:
        component.model = original_model
    credential_cls = Mock(side_effect=AssertionError("provider must not be constructed"))

    async def exercise_api() -> None:
        with patch("reme.components.as_llm.OpenAIAsLLM.credential_cls", credential_cls):
            with pytest.raises(AttributeError, match="does_not_exist"):
                await app.update_component(
                    "as_llm",
                    "default",
                    model=object(),
                    does_not_exist=True,
                )

            assert component._model is original_model
            if injected:
                assert component.model is original_model
            credential_cls.assert_not_called()

    asyncio.run(exercise_api())


def test_replace_component_rebinds_dependents_and_reuses_runtime_model(tmp_path):
    """A live backend replacement switches the wrapper and every bind target."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    old_model = object()
    verified_model = object()

    async def exercise_api() -> None:
        old_component = await app.update_component("as_llm", "default", model=old_model)
        await app.start()
        wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

        replacement = await app.replace_component(
            "as_llm",
            "default",
            config={
                "backend": "dashscope",
                "model": "consumer-injected",
                "credential": {"api_key": ""},
            },
            runtime_updates={"model": verified_model},
        )

        assert isinstance(replacement, DashScopeAsLLM)
        assert replacement.model is verified_model
        assert replacement.is_started is True
        assert old_component.is_started is False
        assert wrapper.as_llm is replacement
        assert app.context.components[ComponentEnum.AS_LLM]["default"] is replacement
        assert app.config.components[ComponentEnum.AS_LLM]["default"].backend == "dashscope"
        assert old_component not in app._started_components
        assert replacement in app._started_components
        await app.close()

    asyncio.run(exercise_api())


def test_replace_component_before_start_preserves_dependency_order(tmp_path):
    """Unresolved bind placeholders continue to resolve during normal startup."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    verified_model = object()

    async def exercise_api() -> None:
        replacement = await app.replace_component(
            "as_llm",
            "default",
            config={
                "backend": "dashscope",
                "model": "consumer-injected",
                "credential": {"api_key": ""},
            },
            runtime_updates={"model": verified_model},
        )
        wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

        assert wrapper.dependencies[0].name == "default"
        await app.start()
        assert wrapper.as_llm is replacement
        assert replacement.is_started is True
        await app.close()

    asyncio.run(exercise_api())


def test_replace_component_flushes_persistent_state_before_start(tmp_path):
    """A replacement loads runtime state that the old generation had not dumped."""
    config = _file_graph_config(str(tmp_path))
    app = ReMe(**config)

    async def exercise_api() -> None:
        await app.start()
        old_graph = app.context.components[ComponentEnum.FILE_GRAPH]["default"]
        await old_graph.upsert_nodes([FileNode(path="memory.md", st_mtime=1.0)])

        replacement = await app.replace_component(
            "file_graph",
            "default",
            config={"backend": "local"},
        )

        assert [node.path for node in await replacement.get_nodes()] == ["memory.md"]
        await app.close()

        restored_app = ReMe(**config)
        await restored_app.start()
        restored_graph = restored_app.context.components[ComponentEnum.FILE_GRAPH]["default"]
        assert [node.path for node in await restored_graph.get_nodes()] == ["memory.md"]
        await restored_app.close()

    asyncio.run(exercise_api())


def test_replace_component_accepts_unhashable_plugin_component(tmp_path):
    """Shutdown-order calculation relies on identity for legal unhashable plugins."""

    class UnhashableAsLLM(BaseAsLLM):
        """Plugin component whose equality contract intentionally disables hashing."""

        component_type = ComponentEnum.AS_LLM
        __hash__ = None

        def __eq__(self, other) -> bool:
            return self is other

    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    app.context.registry.add("unhashable", UnhashableAsLLM, owner="test")

    async def exercise_api() -> None:
        await app.update_component("as_llm", "default", model=object())
        await app.start()

        replacement = await app.replace_component(
            "as_llm",
            "default",
            config={"backend": "unhashable"},
            runtime_updates={"model": object()},
        )

        assert any(component is replacement for component in app._started_components)
        await app.close()
        assert replacement.is_started is False

    asyncio.run(exercise_api())


def test_replace_component_dump_failure_keeps_old_generation(tmp_path):
    """A failed state flush prevents replacement startup and public mutation."""

    class ObservedAsLLM(BaseAsLLM):
        """Replacement backend that records whether startup was attempted."""

        component_type = ComponentEnum.AS_LLM
        start_calls = 0

        async def _start(self) -> None:
            type(self).start_calls += 1

    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    app.context.registry.add("observed", ObservedAsLLM, owner="test")

    async def failing_dump() -> None:
        raise RuntimeError("state flush failed")

    async def exercise_api() -> None:
        old_component = await app.update_component("as_llm", "default", model=object())
        await app.start()
        wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]
        old_component.dump = failing_dump

        with pytest.raises(RuntimeError, match="state flush failed"):
            await app.replace_component(
                "as_llm",
                "default",
                config={"backend": "observed"},
            )

        assert ObservedAsLLM.start_calls == 0
        assert app.context.components[ComponentEnum.AS_LLM]["default"] is old_component
        assert wrapper.as_llm is old_component
        assert old_component.is_started is True
        await app.close()

    asyncio.run(exercise_api())


def test_replace_component_waits_for_application_start(tmp_path):
    """Replacement cannot race the dependency graph while startup is in progress."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    old_component = app.context.components[ComponentEnum.AS_LLM]["default"]
    start_entered = asyncio.Event()
    allow_start = asyncio.Event()

    async def blocking_start() -> None:
        start_entered.set()
        await allow_start.wait()

    old_component._start = blocking_start

    async def exercise_api() -> None:
        start_task = asyncio.create_task(app.start())
        await start_entered.wait()
        replace_task = asyncio.create_task(
            app.replace_component(
                "as_llm",
                "default",
                config={
                    "backend": "dashscope",
                    "model": "consumer-injected",
                    "credential": {"api_key": ""},
                },
                runtime_updates={"model": object()},
            ),
        )
        await asyncio.sleep(0)

        assert replace_task.done() is False
        assert app.context.components[ComponentEnum.AS_LLM]["default"] is old_component

        allow_start.set()
        await start_task
        replacement = await replace_task
        assert replacement.is_started is True
        await app.close()

    asyncio.run(exercise_api())


def test_replace_component_start_failure_keeps_old_generation(tmp_path):
    """Construction/start failures do not expose a partially replaced graph."""

    class BrokenAsLLM(BaseAsLLM):
        """Backend whose startup deterministically fails for rollback tests."""

        component_type = ComponentEnum.AS_LLM

        async def _start(self) -> None:
            raise RuntimeError("replacement failed")

    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    app.context.registry.add("broken", BrokenAsLLM, owner="test")

    async def exercise_api() -> None:
        old_component = await app.update_component("as_llm", "default", model=object())
        await app.start()
        wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

        with pytest.raises(RuntimeError, match="replacement failed"):
            await app.replace_component(
                "as_llm",
                "default",
                config={
                    "backend": "broken",
                    "model": "unused",
                    "credential": {},
                },
            )

        assert app.context.components[ComponentEnum.AS_LLM]["default"] is old_component
        assert app.config.components[ComponentEnum.AS_LLM]["default"].backend == "openai"
        assert wrapper.as_llm is old_component
        assert old_component.is_started is True
        assert old_component in app._started_components
        await app.close()

    asyncio.run(exercise_api())


@pytest.mark.parametrize("operation", ["update", "replace"])
def test_model_injection_does_not_initialize_provider(tmp_path, operation):
    """Attribute validation must not construct a provider before model injection."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    model = object()
    credential_cls = Mock(side_effect=AssertionError("provider must not be constructed"))

    async def run():
        with patch("reme.components.as_llm.OpenAIAsLLM.credential_cls", credential_cls):
            if operation == "update":
                component = await app.update_component("as_llm", "default", model=model)
            else:
                component = await app.replace_component(
                    "as_llm",
                    "default",
                    config={"backend": "openai", "model": "injected"},
                    runtime_updates={"model": model},
                )
            await app.start()
            try:
                assert component.model is model
                credential_cls.assert_not_called()
            finally:
                await app.close()

    asyncio.run(run())
