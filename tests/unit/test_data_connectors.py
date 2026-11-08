"""Resource publication and screen capture without accessing a real desktop."""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.runtime_context import RuntimeContext
from reme.components.screen_capture import ScreenCapture
from reme.config.config_parser import resolve_app_config
from reme.schema import ConnectorResult
from reme.steps.proactive.data_connectors import BaseConnectorStep, ScreenCaptureStep
from reme.utils.async_utils import finish_cleanup
from reme.utils import screen_capture_worker


@pytest.fixture(name="screens")
def _screens(monkeypatch):
    """Two unequal displays and a synthetic encoder, never a system screenshot."""
    state = SimpleNamespace(
        monitors=[
            {"left": 0, "top": 0, "width": 3, "height": 3},
            {"left": 0, "top": 0, "width": 2, "height": 1, "unique_id": "display-a"},
            {"left": 2, "top": 0, "width": 1, "height": 3, "unique_id": "display-b"},
        ],
        captured=[],
        encoded=[],
        threads=[],
        closed=0,
        fail_display=None,
        on_grab=None,
        colors={1: 1, 2: 2},
        fail_encode=False,
    )

    def grab(monitor):
        state.threads.append(threading.get_ident())
        display_id = next(
            index
            for index, value in enumerate(state.monitors)
            if all(value[key] == monitor[key] for key in ("left", "top", "width", "height"))
        )
        if state.on_grab:
            state.on_grab(display_id)
        if display_id == state.fail_display:
            raise RuntimeError("display capture failed")
        state.captured.append(display_id)
        return SimpleNamespace(
            rgb=bytes([state.colors[display_id]]) * (3 * monitor["width"] * monitor["height"]),
            size=(monitor["width"], monitor["height"]),
        )

    @contextmanager
    def capture():
        state.threads.append(threading.get_ident())
        try:
            yield SimpleNamespace(monitors=state.monitors, grab=grab)
        finally:
            state.closed += 1

    from PIL import Image

    save = Image.Image.save

    def encode(image, output, *args, **kwargs):
        if Path(output).parent.name.startswith(".reme-screen-"):
            state.encoded.append(image.size)
            if state.fail_encode:
                raise OSError("encoder failed")
        return save(image, output, *args, **kwargs)

    def displays(instance):
        return [
            {
                "display_id": monitor["unique_id"],
                "stable_id": True,
                "rotation": 0,
                "bounds": [monitor[key] for key in ("left", "top", "width", "height")],
            }
            for monitor in instance.monitors[1:]
        ]

    async def isolated_capture(_service, directory, **_kwargs):
        native = asyncio.create_task(asyncio.to_thread(screen_capture_worker.capture_displays, directory, False))
        try:
            return await asyncio.shield(native)
        except asyncio.CancelledError:
            await finish_cleanup(native)
            raise
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    module = ModuleType("mss")
    module.mss = capture
    module.MSS = capture
    monkeypatch.setitem(sys.modules, "mss", module)
    monkeypatch.setattr(Image.Image, "save", encode)
    monkeypatch.setattr(screen_capture_worker, "_displays", displays)
    monkeypatch.setattr(screen_capture_worker, "_overlays", lambda: [])
    monkeypatch.setattr(ScreenCapture, "capture", isolated_capture)
    return state


def _app(tmp_path, **kwargs):
    app = ApplicationContext(workspace_dir=str(tmp_path), **kwargs)
    app.components["screen_capture"] = {"default": ScreenCapture(app_context=app, ignore_clock=False)}
    return app


def display_token(identifier):
    """Filename tokens depend on stable native identity, not display ordering."""
    return hashlib.sha256(identifier.encode()).hexdigest()[:24]


@pytest.mark.asyncio
async def test_all_displays_return_context_and_original_size_image_paths(tmp_path, screens, monkeypatch):
    """Different display sizes survive, with no visible package until every payload is ready."""
    app = _app(tmp_path, resource_dir="assets", timezone="Pacific/Honolulu")
    instant = datetime(2026, 11, 5, 1, 0, tzinfo=timezone.utc)

    class Clock(datetime):
        """Fix the instant across a local-date boundary."""

        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz)

    monkeypatch.setattr("reme.steps.proactive.data_connectors.base_connector.datetime", Clock)

    def before_grab(_display_id):
        assert not list((tmp_path / "assets").rglob("metadata.json"))

    screens.on_grab = before_grab
    response = await ScreenCaptureStep(app_context=app)()
    paths = response.answer["paths"]
    assert set(response.answer) == {"context", "paths"}
    assert "2026-11-04T15:00:00-10:00" in response.answer["context"]
    assert "2 displays" in response.answer["context"]
    assert paths == [
        f"assets/screen_capture/2026-11-04_15-00-00-000000_display_{display_token('display-a')}.png",
        f"assets/screen_capture/2026-11-04_15-00-00-000000_display_{display_token('display-b')}.png",
    ]
    from PIL import Image

    with Image.open(tmp_path / paths[0]) as first, Image.open(tmp_path / paths[1]) as second:
        assert first.getpixel((0, 0)) == (1, 1, 1)
        assert second.getpixel((0, 0)) == (2, 2, 2)
    assert not list(tmp_path.rglob("metadata.json"))
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
        screens.fail_encode = True
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
async def test_concurrent_collections_do_not_queue_or_overwrite_each_other(tmp_path, screens):
    """One capture owns the native slot; overlapping requests fail without collecting."""
    app = _app(tmp_path)
    results = await asyncio.gather(*(ScreenCaptureStep(app_context=app)() for _ in range(2)), return_exceptions=True)
    failures = [result for result in results if isinstance(result, RuntimeError)]
    assert len(failures) == 1 and "already running" in str(failures[0])
    paths = {path for response in results if not isinstance(response, Exception) for path in response.answer["paths"]}
    assert len(paths) == 2
    assert all((tmp_path / path).is_file() for path in paths)
    assert screens.closed == 1


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
    ["", ".", "../a.png", "/a.png", "a/../../b.png", "a\\b.png", "C:/a.png", "a//b.png"],
)
def test_connector_result_rejects_unsafe_paths(path):
    """Connector paths must be unambiguous relative references."""
    with pytest.raises(ValueError):
        ConnectorResult(context="test", paths=[path])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["invalid_result", "duplicate", "missing", "symlink"])
async def test_base_connector_rejects_invalid_results(tmp_path, kind):
    """References must be distinct, existing files or directories without symlinks."""
    external = tmp_path / "keep.txt"
    external.write_text("preserve", encoding="utf-8")

    class InvalidConnector(BaseConnectorStep):
        """Exercise the shared publication boundary rather than screenshot internals."""

        source = "test"

        def collect(self, directory, created_at):
            if kind == "invalid_result":
                return {"context": "test", "paths": "a.txt"}
            if kind == "symlink":
                (directory / "a.txt").symlink_to(external)
            elif kind != "missing":
                (directory / "a.txt").write_text("payload", encoding="utf-8")
            return ConnectorResult(context="test", paths=["a.txt"] * (2 if kind == "duplicate" else 1))

    with pytest.raises(ValueError):
        await InvalidConnector(app_context=_app(tmp_path))()
    assert external.read_text() == "preserve"
    assert not list((tmp_path / "resource").rglob("*"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_capture_job_is_registered_without_a_resource_catalog(tmp_path, screens):
    """Manual and scheduled jobs use the same connector without discovery or acknowledgements."""
    app = ApplicationContext(**resolve_app_config(config="proactive", log_config=False, workspace_dir=str(tmp_path)))
    app.components["screen_capture"] = {"default": ScreenCapture(app_context=app, ignore_clock=False)}
    assert "file_catalog" not in app.app_config.components
    assert not {"resource_watch_loop", "resource_changes", "resource_processed"} & app.app_config.jobs.keys()
    assert "screen_capture_cron" not in app.app_config.jobs
    tasks = app.app_config.components["pa"]["default"].model_extra["default_tasks"]
    assert next(task for task in tasks if task["id"] == "screen_activity")["cron"] == "*/5 * * * *"
    step_config = app.app_config.jobs["screen_capture"].steps[0].model_dump()
    step_cls = app.registry.get("step", step_config["backend"])
    result = (await step_cls(app_context=app, **step_config)()).answer
    assert all((tmp_path / path).is_file() for path in result["paths"])
    assert screens.captured == [1, 2]


@pytest.mark.asyncio
async def test_text_and_directory_results_preserve_progressive_reading(tmp_path):
    """Text-only results and directory references use the same public schema."""

    class Connector(BaseConnectorStep):
        """Supply a directory that an agent can list before opening individual files."""

        source = "test"

        def collect(self, directory, created_at):
            if self.kwargs.get("text_only"):
                return ConnectorResult(context="Full textual result.")
            (directory / "details").mkdir()
            (directory / "details/report.txt").write_text("Full report.")
            return ConnectorResult(context="Read details as needed.", paths=["details"])

    app = _app(tmp_path)
    result = (await Connector(app_context=app, text_only=True)()).answer
    assert result == {"context": "Full textual result.", "paths": []}
    result = (await Connector(app_context=app)()).answer
    assert result == {"context": "Read details as needed.", "paths": ["resource/test/details"]}
    assert (tmp_path / result["paths"][0] / "report.txt").read_text() == "Full report."


@pytest.mark.asyncio
@pytest.mark.usefixtures("screens")
async def test_publish_failure_rolls_back_only_this_collection(tmp_path, monkeypatch):
    """A failure on the second file preserves unrelated files and removes the first publication."""
    target = tmp_path / "resource/screen_capture"
    target.mkdir(parents=True)
    keep = target / "keep.png"
    keep.write_bytes(b"existing")
    link = Path.hardlink_to

    def fail_second(path, source):
        if path.name.endswith(f"display_{display_token('display-b')}.png"):
            raise OSError("publication failed")
        return link(path, source)

    monkeypatch.setattr(Path, "hardlink_to", fail_second)
    with pytest.raises(OSError, match="publication failed"):
        await ScreenCaptureStep(app_context=_app(tmp_path))()
    assert list(target.iterdir()) == [keep]
    assert keep.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.usefixtures("screens")
async def test_timestamp_collision_preserves_existing_files(tmp_path, monkeypatch, screens):
    """Identical capture timestamps fail explicitly rather than overwriting another capture."""

    class Clock(datetime):
        """Force identical filenames for successive captures."""

        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 7, 12, tzinfo=tz)

    monkeypatch.setattr("reme.steps.proactive.data_connectors.base_connector.datetime", Clock)
    app = _app(tmp_path)
    first = (await ScreenCaptureStep(app_context=app)()).answer
    before = {path: (tmp_path / path).read_bytes() for path in first["paths"]}
    screens.colors[1] = 255
    with pytest.raises(FileExistsError):
        await ScreenCaptureStep(app_context=app)()
    assert {path: (tmp_path / path).read_bytes() for path in first["paths"]} == before
