"""Resource package layout, safe publication and event identity."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import datetime
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from reme.components import ApplicationContext, RuntimeContext
from reme.components.agent_runtime import BaseAgentRuntime
from reme.utils.files.workspace_files import WorkspaceFiles
from reme.schema import ResourceMetadata
from reme.steps.proactive.data_connectors import BaseConnectorStep, ResourceEventStep, ScreenCaptureStep


class PackageConnector(BaseConnectorStep):
    """Write a small payload without using a real device."""

    source = "screenshot"

    def collect(self, directory: Path):
        (directory / "1.png").write_bytes(b"image")
        return [{"path": "1.png"}]


@pytest.fixture(name="displays")
def mock_displays(monkeypatch):
    mss = pytest.importorskip("mss")

    capture = Mock()
    capture.monitors = [{}, {"left": 0, "top": 0, "width": 1, "height": 1}]
    capture.grab.return_value = SimpleNamespace(rgb=b"\x00\x00\x00", size=(1, 1))
    capture.__enter__ = Mock(return_value=capture)
    capture.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(mss, "MSS", Mock(return_value=capture))
    return capture


@pytest.mark.asyncio
async def test_screen_groups_skip_unchanged_pixels_and_preserve_a_return_to_an_old_image(tmp_path, displays):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    expected = [False, True, False, False]
    for pixels, skipped in zip((b"\x00\x00\x00", b"\x00\x00\x00", b"\xff\xff\xff", b"\x00\x00\x00"), expected):
        displays.grab.return_value = SimpleNamespace(rgb=pixels, size=(1, 1))
        step = ScreenCaptureStep(app_context=context)
        response = await step()
        assert (response.answer == {"skipped": "no_new_data"}) is skipped
    assert len(list(tmp_path.glob("resource/screenshot/*/*/metadata.json"))) == 3
    assert len(list(tmp_path.glob("resource/screenshot/*/*"))) == 3


@pytest.mark.asyncio
async def test_screen_geometry_changes_are_new_even_when_pixels_match(tmp_path, displays):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    await ScreenCaptureStep(app_context=context)()
    displays.monitors[1]["left"] = 100
    response = await ScreenCaptureStep(app_context=context)()
    assert "path" in response.answer
    assert len(list(tmp_path.glob("resource/screenshot/*/*/metadata.json"))) == 2


@pytest.mark.asyncio
async def test_concurrent_screen_capture_publishes_identical_pixels_once(tmp_path, displays):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    results = await asyncio.gather(*(ScreenCaptureStep(app_context=context)() for _ in range(2)))
    assert sum(response.answer == {"skipped": "no_new_data"} for response in results) == 1
    assert len(list(tmp_path.glob("resource/screenshot/*/*/metadata.json"))) == 1
    assert displays.grab.call_count == 2


@pytest.mark.parametrize("threshold", [True, "0.99", None, -0.01, 1.01, float("nan"), float("inf")])
def test_screen_similarity_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="similarity_threshold"):
        ScreenCaptureStep(similarity_threshold=threshold)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_pixels,threshold,skipped",
    [(1, 0.99, True), (2, 0.99, False), (3, 0.99, False), (3, 0.98, True), (1, 1, False), (0, 1, True)],
)
async def test_screen_similarity_uses_all_rgb_channels_and_a_strict_threshold(
    tmp_path,
    displays,
    changed_pixels,
    threshold,
    skipped,
):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    displays.monitors[1].update(width=20, height=10)
    displays.grab.return_value = SimpleNamespace(rgb=b"\x00\x00\x00" * 200, size=(20, 10))
    await ScreenCaptureStep(app_context=context, similarity_threshold=threshold)()
    displays.grab.return_value = SimpleNamespace(
        rgb=b"\x00\x00\x01" * changed_pixels + b"\x00\x00\x00" * (200 - changed_pixels),
        size=(20, 10),
    )
    response = await ScreenCaptureStep(app_context=context, similarity_threshold=threshold)()
    assert (response.answer == {"skipped": "no_new_data"}) is skipped
    assert len(list(tmp_path.glob("resource/screenshot/*/*"))) == (1 if skipped else 2)


@pytest.mark.asyncio
async def test_screen_similarity_compares_with_the_last_saved_group(tmp_path, displays):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    displays.monitors[1].update(width=20, height=10)
    for changed_pixels, skipped in ((0, False), (1, True), (2, False)):
        displays.grab.return_value = SimpleNamespace(
            rgb=b"\xff\xff\xff" * changed_pixels + b"\x00\x00\x00" * (200 - changed_pixels),
            size=(20, 10),
        )
        response = await ScreenCaptureStep(app_context=context)()
        assert (response.answer == {"skipped": "no_new_data"}) is skipped
    assert len(list(tmp_path.glob("resource/screenshot/*/*"))) == 2


@pytest.mark.asyncio
async def test_changed_small_display_is_not_diluted_by_a_large_unchanged_display(tmp_path, displays):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    displays.monitors = [
        {},
        {"left": 0, "top": 0, "width": 100, "height": 100},
        {"left": 100, "top": 0, "width": 1, "height": 1},
    ]
    large = SimpleNamespace(rgb=b"\x00\x00\x00" * 10000, size=(100, 100))
    displays.grab.side_effect = [
        large,
        SimpleNamespace(rgb=b"\x00\x00\x00", size=(1, 1)),
        large,
        SimpleNamespace(rgb=b"\xff\xff\xff", size=(1, 1)),
    ]
    await ScreenCaptureStep(app_context=context)()
    response = await ScreenCaptureStep(app_context=context)()
    assert [entry["path"] for entry in response.answer["files"]] == ["1.png", "2.png"]
    assert len(list(tmp_path.glob("resource/screenshot/*/*"))) == 2


@pytest.mark.asyncio
async def test_screen_similarity_preserves_image_size_changes(tmp_path, displays):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    displays.grab.return_value = SimpleNamespace(rgb=b"\x00\x00\x00" * 200, size=(20, 10))
    await ScreenCaptureStep(app_context=context)()
    displays.grab.return_value = SimpleNamespace(rgb=b"\x00\x00\x00" * 200, size=(10, 20))
    response = await ScreenCaptureStep(app_context=context)()
    assert "path" in response.answer


@pytest.mark.asyncio
async def test_screen_similarity_rejects_symlinked_history_and_cleans_new_capture(tmp_path, displays):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    response = await ScreenCaptureStep(app_context=context)()
    previous = tmp_path / response.answer["path"]
    image = previous.parent / "1.png"
    target = tmp_path / "other.png"
    image.rename(target)
    image.symlink_to(target)
    with pytest.raises(ValueError, match="without symlinks"):
        await ScreenCaptureStep(app_context=context)()
    assert previous.is_file() and target.is_file()
    assert len(list(tmp_path.glob("resource/screenshot/*/*"))) == 1
    assert displays.grab.call_count == 2


@pytest.mark.asyncio
async def test_empty_collection_skips_publication_and_the_following_event_step(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    invocation = RuntimeContext()
    step = PackageConnector(app_context=context)
    step.collect = Mock(return_value=[])
    await step(invocation)
    assert invocation["resource"] is None
    response = await ResourceEventStep(app_context=context)(invocation)
    assert response.success and response.answer == {"skipped": "no_new_data"}
    assert not list(tmp_path.glob("resource/*/*/*"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_invalid_collection_result_is_not_treated_as_an_empty_batch(tmp_path):
    step = PackageConnector(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    step.collect = Mock(return_value=None)
    with pytest.raises(ValidationError):
        await step()
    assert not list(tmp_path.glob("resource/*/*/*"))


@pytest.mark.asyncio
async def test_resources_are_written_in_place_before_the_manifest_is_atomically_published(tmp_path, monkeypatch):
    step = PackageConnector(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    collect, replace = step.collect, os.replace
    directories, manifests = [], []

    def collect_in_place(directory):
        assert directory.is_relative_to(tmp_path / "resource/screenshot")
        assert not (directory / "metadata.json").exists()
        directories.append(directory)
        return collect(directory)

    def publish_manifest(source, target):
        assert target == directories[0] / "metadata.json"
        assert not target.exists()
        metadata = ResourceMetadata.model_validate_json(source.read_text())
        assert (target.parent / metadata.files[0].path).read_bytes() == b"image"
        manifests.append(target)
        return replace(source, target)

    monkeypatch.setattr(step, "collect", collect_in_place)
    monkeypatch.setattr("reme.utils.files.workspace_files.os.replace", publish_manifest)
    response = await step()
    assert manifests == [tmp_path / response.answer["path"]]
    assert not list(tmp_path.rglob(".agent-*.tmp"))


@pytest.mark.asyncio
async def test_failed_manifest_publication_removes_the_incomplete_resource_package(tmp_path, monkeypatch):
    step = PackageConnector(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    monkeypatch.setattr(
        "reme.utils.files.workspace_files.os.replace",
        Mock(side_effect=OSError("manifest publication failed")),
    )
    with pytest.raises(OSError, match="manifest publication failed"):
        await step()
    assert not list(tmp_path.glob("resource/*/*/*"))
    assert not list(tmp_path.rglob("metadata.json"))
    assert not list(tmp_path.rglob(".agent-*.tmp"))


@pytest.mark.asyncio
async def test_packages_are_grouped_by_source_and_local_capture_time(tmp_path, monkeypatch):
    first = datetime(2026, 11, 15, 12, 30, 45, 123456, tzinfo=ZoneInfo("Asia/Shanghai"))
    clock = Mock()
    clock.now.return_value = first
    monkeypatch.setattr(
        "reme.steps.proactive.data_connectors.base_connector.uuid4",
        Mock(side_effect=[SimpleNamespace(hex="a8f3c2" + "0" * 26), SimpleNamespace(hex="b9e4d3" + "0" * 26)]),
    )
    monkeypatch.setattr("reme.steps.proactive.data_connectors.base_connector.datetime", clock)
    step = PackageConnector(app_context=ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai"))
    responses = [await step(), await step()]
    assert [response.answer["path"] for response in responses] == [
        f"resource/screenshot/2026-11-15/123045_{suffix}/metadata.json" for suffix in ("a8f3c2", "b9e4d3")
    ]
    for response in responses:
        manifest = tmp_path / response.answer["path"]
        metadata = ResourceMetadata.model_validate_json(manifest.read_text())
        assert metadata.source == ScreenCaptureStep.source == "screenshot"
        assert metadata.files[0].path == "1.png"
        assert (manifest.parent / metadata.files[0].path).read_bytes() == b"image"
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["", "../outside", "/outside", "a/b", "a\\b", ".."])
async def test_invalid_source_fails_before_collection(tmp_path, source):
    step = PackageConnector(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    step.source = source
    step.collect = Mock(side_effect=AssertionError("must not collect"))
    with pytest.raises(ValueError, match="resource source"):
        await step()
    step.collect.assert_not_called()
    assert not (tmp_path / "resource").exists()


@pytest.mark.asyncio
async def test_event_ids_include_source_and_date_and_are_retry_stable(tmp_path):
    runtime = Mock(spec=BaseAgentRuntime)
    runtime.files = WorkspaceFiles(tmp_path)
    runtime.definition = AsyncMock(return_value=SimpleNamespace(cron=None))
    runtime.append = AsyncMock(return_value={})
    step = ResourceEventStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), agent_runtime=runtime)
    paths = [
        "resource/screenshot/2026-11-15/123045_123456",
        "resource/screenshot/2026-11-16/123045_123456",
        "resource/web_extract/2026-11-15/123045_123456",
    ]
    for package in [*paths, paths[0]]:
        directory = tmp_path / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "1.png").write_bytes(b"image")
        metadata = ResourceMetadata(
            dataclass="PackageConnector",
            source=package.split("/")[1],
            created_at="2026-11-15T12:30:45+08:00",
            modified_at="2026-11-15T12:30:45+08:00",
            description="Captured resource",
            files=[{"path": "1.png"}],
        )
        manifest = f"{package}/metadata.json"
        (tmp_path / manifest).write_text(metadata.model_dump_json())
        await step(agent_name="observer", resource={"path": manifest})
        event = runtime.append.call_args.args[1]
        assert event.paths == [manifest, f"{package}/1.png"]
    ids = [call.args[1].id for call in runtime.append.call_args_list]
    assert len(set(ids[:3])) == 3
    assert ids[0] == ids[-1]


@pytest.mark.asyncio
async def test_identical_package_names_never_overwrite_a_published_package(tmp_path, monkeypatch):
    clock = Mock()
    clock.now.return_value = datetime(2026, 11, 15, 12, 30, 45, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(
        "reme.steps.proactive.data_connectors.base_connector.uuid4",
        Mock(return_value=SimpleNamespace(hex="a8f3c2" + "0" * 26)),
    )
    monkeypatch.setattr("reme.steps.proactive.data_connectors.base_connector.datetime", clock)
    step = PackageConnector(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step()
    manifest = tmp_path / response.answer["path"]
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        await step()
    assert manifest.read_bytes() == original
    assert (manifest.parent / "1.png").read_bytes() == b"image"
    assert not list(tmp_path.glob(".reme-resource-*"))
