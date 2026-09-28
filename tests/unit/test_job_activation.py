"""Application-level activation, including service and lifecycle boundaries."""

# pylint: disable=protected-access,missing-function-docstring,missing-class-docstring

import asyncio
import copy
import multiprocessing
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from watchfiles import Change

from reme.application import Application
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.agent_wrapper.codex_mcp_server import _prepare_config
from reme.components.component_registry import create_application_registry
from reme.components.job import BackgroundJob, BaseJob, CronJob, StreamJob
from reme.config import parse_kwargs, resolve_app_config
from reme.plugin import Backend, Plugin, PluginManager
from reme.schema.application_config import JobConfig
from reme.steps.base_step import BaseStep


def _app(path, jobs, **kwargs):
    return Application(
        workspace_dir=str(path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service=kwargs.pop("service", {"backend": "cli"}),
        jobs=jobs,
        **kwargs,
    )


class _WaitingJob(BackgroundJob):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.calls = 0

    async def __call__(self, **kwargs):
        self.calls += 1
        self.entered.set()
        try:
            await self._stop_event.wait()
        finally:
            self.cleaned.set()


class _CallingStep(BaseStep):
    async def execute(self):
        return await self.run_job("off")


class _ToolWrapper(BaseAgentWrapper):
    async def reply(self, inputs, **kwargs):
        raise AssertionError("no model call expected")


@pytest.fixture(name="job_registry")
def _job_registry(monkeypatch):
    registry = create_application_registry()
    registry.register(_WaitingJob, "activation_wait")
    registry.register(_CallingStep, "activation_call")
    monkeypatch.setattr("reme.plugin.create_application_registry", registry.copy)
    return registry


@pytest.mark.parametrize("value", [None, True])
def test_enabled_defaults_and_explicit_true(value):
    config = JobConfig(**({} if value is None else {"enabled": value}))
    assert config.enabled is True


@pytest.mark.parametrize("value", ["not-a-bool", None, [], 2])
def test_invalid_activation_is_rejected(value):
    with pytest.raises(ValidationError):
        JobConfig(enabled=value)


def test_yaml_and_cli_overrides_preserve_jobs(tmp_path):
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        "jobs:\n  selected:\n    backend: base\n    enabled: false\n"
        "    description: preserved\n  other:\n    backend: base\n",
        encoding="utf-8",
    )
    loaded = resolve_app_config(config=str(config_path), log_config=False)
    assert JobConfig(**loaded["jobs"]["selected"]).enabled is False
    config = resolve_app_config(
        config=str(config_path),
        log_config=False,
        **parse_kwargs("jobs.other.enabled=false"),
    )
    app = _app(tmp_path / "workspace", config["jobs"])
    assert not app.context.jobs
    assert app.config.jobs["selected"].description == "preserved"
    assert set(app.config.jobs) == {"selected", "other"}
    defaults = resolve_app_config(log_config=False, **parse_kwargs("jobs.dream_cron.enabled=false"))
    assert defaults["jobs"]["dream_cron"]["enabled"] is False
    assert defaults["jobs"]["dream_cron"]["steps"]
    assert "index_update_loop" in defaults["jobs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("job_class", [BaseJob, StreamJob, BackgroundJob, CronJob])
async def test_disabled_jobs_are_never_constructed(tmp_path, job_registry, job_class):
    class MustNotConstruct(job_class):
        def __init__(self, **kwargs):
            raise AssertionError("disabled constructor was called")

    job_registry.register(MustNotConstruct, "activation_forbidden")
    config = {"off": {"backend": "activation_forbidden", "enabled": False}}
    original = copy.deepcopy(config)
    app = _app(tmp_path, config)
    await app.start()
    try:
        assert app.context.jobs == {}
        assert app._started_components == []
        with pytest.raises(ValueError, match="Job 'off' is disabled"):
            await app.run_job("off")
        with pytest.raises(ValueError, match="Job 'off' is disabled"):
            await anext(app.run_stream_job("off"))
    finally:
        await app.close()
    assert config == original


def test_disabled_backend_and_steps_are_not_resolved(tmp_path):
    app = _app(
        tmp_path,
        {
            "off": {
                "backend": "not_installed",
                "enabled": False,
                "steps": [{"backend": "also_not_installed"}],
            },
        },
    )
    assert "off" not in app.context.jobs
    with pytest.raises(ValidationError):
        _app(tmp_path, {"off": {"enabled": False, "steps": "invalid"}})


@pytest.mark.asyncio
async def test_enabled_jobs_preserve_kwargs_and_unknown_name_errors(tmp_path):
    app = _app(tmp_path, {"on": {"backend": "base", "marker": "kept"}})
    async with app:
        job = app.context.jobs["on"]
        assert job.kwargs["marker"] == "kept"
        assert "enabled" not in job.kwargs
        assert (await app.run_job("on")).success
        with pytest.raises(KeyError, match="not found"):
            await app.run_job("unknown")
        with pytest.raises(KeyError, match="not found"):
            await anext(app.run_stream_job("unknown"))


@pytest.mark.asyncio
async def test_activation_is_per_application_and_fixed_at_construction(tmp_path):
    config = {"selected": {"backend": "base", "enabled": False}}
    off = _app(tmp_path / "off", config)
    config["selected"]["enabled"] = True
    on = _app(tmp_path / "on", config)
    off.config.jobs["selected"].enabled = True
    on.config.jobs["selected"].enabled = False
    async with off, on:
        with pytest.raises(ValueError, match="disabled"):
            await off.run_job("selected")
        assert (await on.run_job("selected")).success


@pytest.mark.asyncio
async def test_plugin_default_can_be_disabled(tmp_path, monkeypatch):
    manager = PluginManager(
        [
            Plugin(
                name="test",
                backends=(Backend("plugin_wait", _WaitingJob),),
                config={"jobs": {"watch": {"backend": "plugin_wait"}}},
            ),
        ],
    )
    monkeypatch.setattr(PluginManager, "discover", classmethod(lambda cls, specs: manager))
    app = _app(tmp_path, {"watch": {"enabled": False}}, plugins=["test"])
    async with app:
        assert app.config.jobs["watch"].backend == "plugin_wait"
        assert not app.context.jobs


@pytest.mark.asyncio
async def test_background_cleanup_and_idempotence(tmp_path, job_registry):
    del job_registry
    app = _app(
        tmp_path,
        {
            "on": {"backend": "activation_wait"},
            "off": {"backend": "activation_wait", "enabled": False},
        },
    )
    await app.start()
    job = app.context.jobs["on"]
    task = job._task
    try:
        await asyncio.wait_for(job.entered.wait(), 5)
        await app.start()
        assert job.calls == 1
        assert job._task is task
    finally:
        await app.close()
    assert job.cleaned.is_set()
    assert task.done()
    assert job._task is None
    assert not app._started_components


@pytest.mark.asyncio
async def test_startup_failure_rolls_back_started_job(tmp_path, job_registry):
    class FailingCron(CronJob):
        async def _start(self):
            await asyncio.wait_for(self.app_context.jobs["watch"].entered.wait(), 5)
            raise RuntimeError("later startup failed")

    job_registry.register(FailingCron, "activation_fail")
    app = _app(
        tmp_path,
        {
            "watch": {"backend": "activation_wait"},
            "fail": {"backend": "activation_fail", "cron": "* * * * *"},
        },
    )
    job = app.context.jobs["watch"]
    with pytest.raises(RuntimeError, match="later startup failed"):
        await app.start()
    assert job.cleaned.is_set()
    assert job._task is None
    assert not app._started_components


@pytest.mark.asyncio
async def test_cron_wait_is_closed(tmp_path):
    app = _app(tmp_path, {"cron": {"backend": "cron", "cron": "0 0 1 1 *"}})
    cron = app.context.jobs["cron"]
    entered = asyncio.Event()
    original_wait = cron._wait_or_stop

    async def wait(delay):
        entered.set()
        await original_wait(delay)

    cron._wait_or_stop = wait
    await app.start()
    task = cron._task
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        await app.close()
    assert task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("service_backend", ["http", "mcp"])
async def test_service_registration_and_disabled_invocation(tmp_path, service_backend):
    app = _app(
        tmp_path,
        {
            "on": {"backend": "base"},
            "off": {"backend": "base", "enabled": False},
            "private": {"backend": "base", "enable_serve": False},
        },
        service={"backend": service_backend, "web_enabled": False},
    )
    service = app.context.service
    service.build_service(app)
    service.add_jobs(app)
    async with app:
        assert (await app.run_job("private")).success
        server = service.mcp_server if service_backend == "http" else service.service
        async with Client(server) as client:
            assert {tool.name for tool in await client.list_tools()} == {"on"}
            with pytest.raises(ToolError):
                await client.call_tool("off", {})
        if service_backend == "http":
            assert set(service.service.openapi()["paths"]) == {"/on"}
            async with AsyncClient(transport=ASGITransport(app=service.service), base_url="http://test") as client:
                assert (await client.post("/off", json={})).status_code == 404
                assert (await client.post("/on", json={})).status_code == 200


@pytest.mark.parametrize("backend", ["http", "mcp"])
def test_service_allowlist_cannot_enable_disabled_job(tmp_path, backend):
    app = _app(
        tmp_path,
        {"off": {"backend": "base", "enabled": False}},
        service={"backend": backend, "jobs": ["off"]},
    )
    service = app.context.service
    service.add_job = Mock()
    with pytest.raises(ValueError, match="disabled.*off"):
        service.add_jobs(app)
    service.add_job.assert_not_called()


def test_cli_rejects_disabled_target_before_start(tmp_path, capsys):
    app = _app(
        tmp_path,
        {"off": {"backend": "base", "enabled": False}},
        service={"backend": "cli", "job": "off"},
    )
    app.start = AsyncMock()
    with pytest.raises(SystemExit) as error:
        app.run_app()
    assert error.value.code == 1
    app.start.assert_not_awaited()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Job 'off' is disabled" in captured.err


def test_codex_bridge_cannot_reenable_disabled_job(tmp_path):
    prepared = _prepare_config({"jobs": {"off": {"backend": "base", "enabled": False}}}, ["off"])
    assert prepared["jobs"]["off"]["enable_serve"] is True
    app = _app(tmp_path, prepared["jobs"], service=prepared["service"])
    with pytest.raises(ValueError, match="disabled.*off"):
        app.context.service.add_jobs(app)


@pytest.mark.asyncio
async def test_internal_consumers_reject_disabled_jobs(tmp_path, job_registry):
    del job_registry
    app = _app(
        tmp_path,
        {
            "off": {"backend": "base", "enabled": False},
            "caller": {"backend": "base", "steps": [{"backend": "activation_call"}]},
        },
    )
    step = _CallingStep(app_context=app.context)
    with pytest.raises(ValueError, match="Job 'off' is disabled"):
        await step.run_job("off")
    wrapper = _ToolWrapper(app_context=app.context)
    with pytest.raises(ValueError, match="Job 'off' is disabled"):
        wrapper._resolve_job_tools(["off"])
    async with app:
        response = await asyncio.wait_for(app.run_job("caller"), 5)
        assert not response.success
        assert "Job 'off' is disabled" in response.answer


@pytest.mark.asyncio
async def test_local_index_initialization_and_refresh_survive_other_disabled_jobs(tmp_path, monkeypatch):
    """Keep real scan/update/search; supply deterministic filesystem notifications."""
    defaults = resolve_app_config(log_config=False)
    jobs = {
        "index_update_loop": defaults["jobs"]["index_update_loop"],
        "dream_cron": {**defaults["jobs"]["dream_cron"], "enabled": False},
        "search": {"backend": "base", "steps": [{"backend": "bm25_search_step"}]},
    }
    component_types = {"tokenizer", "keyword_index", "file_graph", "file_chunker", "file_store", "tag_index"}
    components = {name: value for name, value in defaults["components"].items() if name in component_types}
    ready = asyncio.Event()
    changed = asyncio.Event()
    refreshed = asyncio.Event()
    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "initial.md").write_text("# Initial\n\nOrchid baseline memory.\n", encoding="utf-8")
    fresh_path = daily / "fresh.md"

    async def watch(*_paths, stop_event, **_kwargs):
        ready.set()
        await changed.wait()
        yield {(Change.added, str(fresh_path))}
        refreshed.set()
        await stop_event.wait()

    monkeypatch.setattr("reme.steps.index.watch_changes.awatch", watch)
    app = _app(tmp_path, jobs, components=components)
    await app.start()
    task = app.context.jobs["index_update_loop"]._task
    try:
        await asyncio.wait_for(ready.wait(), 10)
        initial = await app.run_job("search", query="orchid")
        assert initial.success and initial.metadata["results"]
        fresh_path.write_text("# Fresh\n\nSaffron newly indexed memory.\n", encoding="utf-8")
        changed.set()
        await asyncio.wait_for(refreshed.wait(), 10)
        fresh = await app.run_job("search", query="saffron")
        assert fresh.success
        assert any("fresh.md" in str(result) for result in fresh.metadata["results"])
        assert "dream_cron" not in app.context.jobs
    finally:
        changed.set()
        await app.close()
    assert task.done()


def _process_worker(workspace, enabled, events, stop):
    """Spawn-safe worker: exercise the real background task without external services."""
    from unittest.mock import patch

    class ProcessJob(_WaitingJob):
        async def __call__(self, **kwargs):
            events.put((enabled, "run"))
            await super().__call__(**kwargs)

    registry = create_application_registry()
    registry.register(ProcessJob, "process_wait")
    with patch("reme.plugin.create_application_registry", return_value=registry):
        app = _app(workspace, {"probe": {"backend": "process_wait", "enabled": enabled}})

    async def run():
        async with app:
            if enabled:
                await asyncio.wait_for(app.context.jobs["probe"].entered.wait(), 5)
            events.put((enabled, "ready"))
            await asyncio.to_thread(stop.wait, 20)
        events.put((enabled, "closed"))

    asyncio.run(run())


def test_three_processes_only_enabled_job_runs(tmp_path):
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    stop = context.Event()
    processes = [
        context.Process(target=_process_worker, args=(str(tmp_path / str(i)), i == 2, events, stop)) for i in range(3)
    ]
    seen = []
    try:
        for process in processes:
            process.start()
        while sum(event == "ready" for _, event in seen) < 3:
            seen.append(events.get(timeout=30))
        assert [enabled for enabled, event in seen if event == "run"] == [True]
        stop.set()
        while sum(event == "closed" for _, event in seen) < 3:
            seen.append(events.get(timeout=30))
        assert [enabled for enabled, event in seen if event == "run"] == [True]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        stop.set()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        events.close()
        events.join_thread()
