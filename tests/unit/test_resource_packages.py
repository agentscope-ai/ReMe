"""Resource package layout, safe publication and event identity."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest

from reme.components import ApplicationContext
from reme.components.agent_runtime import BaseAgentRuntime
from reme.components.agent_runtime._internal.files import AgentFiles
from reme.schema import ResourceMetadata
from reme.steps.proactive.data_connectors import BaseConnectorStep, ResourceEventStep, ScreenCaptureStep
from reme.steps.proactive.tools.web_extract import WebExtractStep


class PackageConnector(BaseConnectorStep):
    """Write a small payload without using a real device."""

    source = "screenshot"

    def collect(self, directory: Path):
        (directory / "1.png").write_bytes(b"image")
        return [{"path": "1.png"}]


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
    assert WebExtractStep.source == "web_extract"
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
    runtime.files = AgentFiles(tmp_path)
    runtime.definition = AsyncMock(return_value=SimpleNamespace(trigger_type="event"))
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
        await step(agent_id="observer", resource={"path": manifest})
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
