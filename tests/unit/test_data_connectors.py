"""Resource publication and screen capture without accessing a real desktop."""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from watchfiles import Change

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import resolve_app_config
from reme.schema import ResourceFile, ResourceMetadata
from reme.steps.data_connectors import BaseConnectorStep, ScreenCaptureStep
from reme.steps.index.watch_changes import WatchChangesStep
from reme.utils.file_watch import collect_existing, match_file


@pytest.fixture(name="screens")
def _screens(monkeypatch):
    """Two unequal displays and a synthetic encoder, never a system screenshot."""
    state = SimpleNamespace(
        monitors=[
            {"left": 0, "top": 0, "width": 3, "height": 3},
            {"left": 0, "top": 0, "width": 2, "height": 1},
            {"left": 2, "top": 0, "width": 1, "height": 3},
        ],
        captured=[],
        encoded=[],
        threads=[],
        closed=0,
        fail_display=None,
        on_grab=None,
    )

    def grab(monitor):
        state.threads.append(threading.get_ident())
        display_id = state.monitors.index(monitor)
        if state.on_grab:
            state.on_grab(display_id)
        if display_id == state.fail_display:
            raise RuntimeError("display capture failed")
        state.captured.append(display_id)
        return SimpleNamespace(rgb=bytes([display_id]) * 3, size=(monitor["width"], monitor["height"]))

    @contextmanager
    def capture():
        state.threads.append(threading.get_ident())
        try:
            yield SimpleNamespace(monitors=state.monitors, grab=grab)
        finally:
            state.closed += 1

    def encode(rgb, size, output):
        state.encoded.append(size)
        Path(output).write_bytes(rgb)

    module = ModuleType("mss")
    module.mss = capture
    module.tools = ModuleType("mss.tools")
    module.tools.to_png = encode
    monkeypatch.setitem(sys.modules, "mss", module)
    monkeypatch.setitem(sys.modules, "mss.tools", module.tools)
    return state


def _app(tmp_path, **kwargs):
    return ApplicationContext(workspace_dir=str(tmp_path), **kwargs)


@pytest.mark.asyncio
async def test_all_displays_publish_one_minimal_package(tmp_path, screens, monkeypatch):
    """Different display sizes survive, with no visible package until every payload is ready."""
    app = _app(tmp_path, resource_dir="assets", timezone="Pacific/Honolulu")
    instant = datetime(2026, 11, 5, 1, 0, tzinfo=timezone.utc)

    class Clock(datetime):
        """Fix the instant across a local-date boundary."""

        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz)

    monkeypatch.setattr("reme.steps.data_connectors.base_connector.datetime", Clock)

    def before_grab(_display_id):
        assert not list((tmp_path / "assets").rglob("metadata.json"))

    screens.on_grab = before_grab
    response = await ScreenCaptureStep(app_context=app)()
    manifest = tmp_path / response.answer["path"]
    metadata = ResourceMetadata.model_validate_json(manifest.read_text(encoding="utf-8"))
    assert manifest.parent.parent == tmp_path / "assets/2026-11-04"
    assert len(manifest.parent.name) == 32
    int(manifest.parent.name, 16)
    assert metadata.created_at.isoformat() == "2026-11-04T15:00:00-10:00"
    assert metadata.modified_at == metadata.created_at
    assert metadata.dataclass == "ScreenCaptureStep"
    assert metadata.source == "screen"
    assert metadata.description
    assert json.loads(manifest.read_text())["files"] == [{"path": "1.png"}, {"path": "2.png"}]
    assert (manifest.parent / "1.png").read_bytes() == bytes([1]) * 3
    assert (manifest.parent / "2.png").read_bytes() == bytes([2]) * 3
    assert screens.captured == [1, 2]
    assert screens.encoded == [(2, 1), (1, 3)]
    assert len(set(screens.threads)) == 1
    assert screens.threads[0] != threading.get_ident()
    assert screens.closed == 1
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["second_display", "no_displays", "encoder", "missing_mss"])
async def test_failed_capture_publishes_nothing(tmp_path, screens, monkeypatch, failure):
    """A partially captured or unencodable collection never becomes a resource."""
    if failure == "second_display":
        screens.fail_display = 2
    elif failure == "no_displays":
        screens.monitors = screens.monitors[:1]
    elif failure == "encoder":
        monkeypatch.setattr(sys.modules["mss.tools"], "to_png", lambda *_args, **_kwargs: None)
    else:
        monkeypatch.setitem(sys.modules, "mss", None)
    with pytest.raises((RuntimeError, ValueError)):
        await ScreenCaptureStep(app_context=_app(tmp_path))()
    assert not list((tmp_path / "resource").rglob("*"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_cancelled_capture_finishes_cleanup_without_publication(tmp_path, screens):
    """Cancellation waits for native I/O to exit, then discards the staged files."""
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def block_capture(_display_id):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)

    screens.on_grab = block_capture
    task = asyncio.create_task(ScreenCaptureStep(app_context=_app(tmp_path))())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert screens.closed == 1
    assert not list((tmp_path / "resource").rglob("*"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["cancel", "stop"])
async def test_capture_stopped_during_destination_preparation_publishes_nothing(tmp_path, screens, monkeypatch, signal):
    """Cancellation or a stop during parent creation must still discard the complete staged package."""
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    mkdir = Path.mkdir

    def block_mkdir(path, *args, **kwargs):
        if path.parent == tmp_path / "resource":
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", block_mkdir)
    context = RuntimeContext(stop_event=asyncio.Event())
    task = asyncio.create_task(ScreenCaptureStep(app_context=_app(tmp_path))(context))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        if signal == "cancel":
            task.cancel()
            await asyncio.sleep(0)
        else:
            context.stop_event.set()
    finally:
        release.set()
    error = asyncio.CancelledError if signal == "cancel" else RuntimeError
    with pytest.raises(error):
        await asyncio.wait_for(task, timeout=5)
    assert screens.captured == [1, 2]
    assert screens.closed == 1
    assert not list((tmp_path / "resource").glob("*/*"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_concurrent_collections_do_not_overwrite_each_other(tmp_path, screens):
    """Fresh invocation-scoped collectors publish distinct UUID directories."""
    app = _app(tmp_path)
    results = await asyncio.gather(*(ScreenCaptureStep(app_context=app)() for _ in range(2)))
    paths = {response.answer["path"] for response in results}
    assert len(paths) == 2
    assert all((tmp_path / path).is_file() for path in paths)
    assert screens.closed == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [[], ["daily"], ["../outside"]])
async def test_capture_respects_allowed_paths_before_collecting(tmp_path, screens, scope):
    """An invalid or unrelated request scope cannot trigger capture or writes."""
    with pytest.raises(PermissionError):
        await ScreenCaptureStep(app_context=_app(tmp_path))(_allowed_paths=scope)
    assert screens.captured == []
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_resource_directory_symlink_is_rejected(tmp_path, screens):
    """Configured output aliases cannot redirect a collector outside its resource root."""
    (tmp_path / "archive").mkdir()
    (tmp_path / "resource").symlink_to(tmp_path / "archive", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        await ScreenCaptureStep(app_context=_app(tmp_path))()
    assert screens.captured == []
    assert not list((tmp_path / "archive").iterdir())


@pytest.mark.parametrize(
    "path",
    ["", ".", "../a.png", "/a.png", "a/../../b.png", "a\\b.png", "C:/a.png", "metadata.json"],
)
def test_resource_file_rejects_non_payload_paths(path):
    """A manifest cannot address files outside its package or list itself."""
    with pytest.raises(ValueError):
        ResourceFile(path=path)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["empty", "duplicate", "missing", "symlink"])
async def test_base_connector_rejects_incomplete_manifests(tmp_path, kind):
    """Generic connectors must return a nonempty, distinct, existing payload list."""
    external = tmp_path / "keep.txt"
    external.write_text("preserve", encoding="utf-8")

    class InvalidConnector(BaseConnectorStep):
        """Exercise the shared publication boundary rather than screenshot internals."""

        source = "test"

        def collect(self, directory):
            if kind == "empty":
                return []
            if kind == "symlink":
                (directory / "a.txt").symlink_to(external)
            elif kind != "missing":
                (directory / "a.txt").write_text("payload", encoding="utf-8")
            return [{"path": "a.txt"}] * (2 if kind == "duplicate" else 1)

    with pytest.raises(ValueError):
        await InvalidConnector(app_context=_app(tmp_path))()
    assert external.read_text() == "preserve"
    assert not list((tmp_path / "resource").rglob("*"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_proactive_catalog_and_watcher_only_match_manifests(tmp_path, screens):
    """Capture -> changes -> processed operates once per package and never per image."""
    app = ApplicationContext(**resolve_app_config(config="proactive", log_config=False, workspace_dir=str(tmp_path)))
    config = app.app_config.components["file_catalog"]["resource"].model_dump()
    catalog = LocalFileCatalog(name="resource", app_context=app, **config)
    app.components["file_catalog"] = {"resource": catalog}
    job = app.app_config.jobs["screen_capture_cron"]
    assert job.backend == "cron"
    assert job.model_extra["cron"] == "*/15 * * * *"
    step_config = job.steps[0].model_dump()
    step_cls = app.registry.get("step", step_config["backend"])
    response = await step_cls(app_context=app, **step_config)()
    assert screens.captured == [1, 2]
    manifest = tmp_path / response.answer["path"]
    for filename in ("other.json", "prefix.metadata.json", "METADATA.JSON"):
        (manifest.parent / filename).write_text("{}", encoding="utf-8")
    watcher = WatchChangesStep(app_context=app, file_catalog="resource")
    watcher.context = RuntimeContext()
    rules = watcher._get_watch_rules()  # pylint: disable=protected-access
    watcher._rules = rules  # pylint: disable=protected-access
    assert list(collect_existing(rules, recursive=True)) == [str(manifest)]
    for file in manifest.parent.iterdir():
        expected = file == manifest
        assert match_file(str(file), rules) is expected
        assert watcher._filter(Change.added, str(file)) is expected  # pylint: disable=protected-access
    changes = await catalog.changes()
    assert changes == [response.answer["path"]]
    assert await catalog.processed(changes) is True
    (manifest.parent / "1.png").write_bytes(b"edited without updating the manifest")
    assert await catalog.changes() == []
    manifest.unlink()
    assert await catalog.changes() == []
    assert await catalog.processed(changes) is False
