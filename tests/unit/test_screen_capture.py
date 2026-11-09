"""Stable display comparison, bounded native processes, and preserved source pixels."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import hashlib
import sys

import numpy as np
from PIL import Image
import pytest

from reme.components import ApplicationContext, ScreenCapture
from reme.components.runtime_context import RuntimeContext
from reme.steps.proactive.data_connectors import ScreenCaptureStep
from reme.utils.screen_compare import compare_screens


class Captures(ScreenCapture):
    """Replace the native process boundary while exercising real PNG publication."""

    def __init__(self, **kwargs):
        super().__init__(ignore_clock=False, **kwargs)
        self.frames = [
            ("display-A", np.zeros((96, 160, 3), dtype=np.uint8)),
            ("display-B", np.zeros((96, 160, 3), dtype=np.uint8)),
        ]
        self.regions = []
        self.stable = True
        self.calls = 0

    async def capture(self, directory, **_kwargs):
        self.calls += 1
        result = []
        for identifier, pixels in self.frames:
            name = f"display_{hashlib.sha256(identifier.encode()).hexdigest()[:24]}.png"
            image = Image.fromarray(pixels)
            image.save(directory / name)
            result.append(
                {
                    "display_id": identifier,
                    "stable_id": self.stable,
                    "bounds": [0, 0, image.width, image.height],
                    "pixel_size": list(image.size),
                    "rotation": 0,
                    "path": name,
                    "ignored_regions": self.regions,
                },
            )
        return result


def setup(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    capture = Captures(app_context=context)
    context.components["screen_capture"] = {"default": capture}
    return context, capture


@pytest.mark.asyncio
async def test_display_reordering_is_unchanged_and_only_changed_screen_is_published(tmp_path):
    context, capture = setup(tmp_path)
    first = await ScreenCaptureStep(app_context=context)()
    capture.frames.reverse()
    same = await ScreenCaptureStep(app_context=context)()
    assert same.answer["paths"] == [] and "一致" in same.answer["context"]
    assert len(list((tmp_path / "resource/screen_capture").glob("*.png"))) == 2
    capture.frames[0][1][20:35, 30:45] = 255
    changed = await ScreenCaptureStep(app_context=context)()
    assert len(changed.answer["paths"]) == 1
    frames = changed.metadata["screen_capture"]
    assert next(frame for frame in frames if frame["display_id"] == "display-A")["path"] in first.answer["paths"]
    assert next(frame for frame in frames if frame["display_id"] == "display-B")["path"] in changed.answer["paths"]
    for name in first.answer["paths"] + changed.answer["paths"]:
        with Image.open(tmp_path / name) as image:
            assert image.size == (160, 96) and "reme_screen_capture" in image.info


@pytest.mark.asyncio
async def test_exclusions_follow_old_and_new_companion_positions(tmp_path):
    context, capture = setup(tmp_path)
    capture.regions = [(20, 20, 60, 60)]
    capture.frames[0][1][20:60, 20:60] = 50
    await ScreenCaptureStep(app_context=context)()
    capture.regions = [(80, 20, 120, 60)]
    capture.frames[0][1][:] = 0
    capture.frames[0][1][20:60, 80:120] = 200
    assert (await ScreenCaptureStep(app_context=context)()).answer["paths"] == []
    capture.frames[0][1][75:90, 20:40] = 255
    assert len((await ScreenCaptureStep(app_context=context)()).answer["paths"]) == 1


@pytest.mark.asyncio
async def test_small_accumulated_changes_compare_to_last_accepted_frame(tmp_path):
    context, capture = setup(tmp_path)
    await ScreenCaptureStep(app_context=context)()
    capture.frames[0][1][0, :5] = 255
    assert (await ScreenCaptureStep(app_context=context)()).answer["paths"] == []
    capture.frames[0][1][0, :10] = 255
    assert len((await ScreenCaptureStep(app_context=context)()).answer["paths"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["unknown_id", "missing_baseline", "resolution", "corrupt_baseline"])
async def test_unknown_or_unusable_baselines_never_silently_skip(tmp_path, condition):
    context, capture = setup(tmp_path)
    first = await ScreenCaptureStep(app_context=context)()
    if condition == "unknown_id":
        capture.stable = False
    elif condition == "missing_baseline":
        for name in first.answer["paths"]:
            (tmp_path / name).unlink()
    elif condition == "corrupt_baseline":
        for name in first.answer["paths"]:
            (tmp_path / name).write_bytes(b"broken image")
    else:
        capture.frames = [(identifier, np.zeros((100, 160, 3), dtype=np.uint8)) for identifier, _ in capture.frames]
    assert len((await ScreenCaptureStep(app_context=context)()).answer["paths"]) == 2


@pytest.mark.asyncio
async def test_explicit_comparison_baseline_does_not_advance_manual_baseline(tmp_path):
    context, capture = setup(tmp_path)
    first = await ScreenCaptureStep(app_context=context)()
    capture.frames[0][1][20:40, 20:40] = 255
    one = await ScreenCaptureStep(app_context=context)(RuntimeContext(_screen_baseline=[], _screen_scope="task:screen"))
    two = await ScreenCaptureStep(app_context=context)(RuntimeContext(_screen_baseline=[], _screen_scope="task:screen"))
    assert len(one.answer["paths"]) == len(two.answer["paths"]) == 2
    assert len((await ScreenCaptureStep(app_context=context)()).answer["paths"]) == 1
    baseline = one.metadata["screen_capture"]
    same = await ScreenCaptureStep(app_context=context)(
        RuntimeContext(_screen_baseline=baseline, _screen_scope="task:screen"),
    )
    assert same.answer["paths"] == []
    assert all((tmp_path / name).is_file() for name in first.answer["paths"])


@pytest.mark.asyncio
async def test_each_tool_call_captures_and_compares_current_frames(tmp_path):
    context, capture = setup(tmp_path)
    first = await ScreenCaptureStep(app_context=context)()
    second = await ScreenCaptureStep(app_context=context)()
    assert capture.calls == 2 and second.answer["paths"] == []
    (tmp_path / first.answer["paths"][0]).unlink()
    refreshed = await ScreenCaptureStep(app_context=context)()
    assert capture.calls == 3 and refreshed.answer["paths"]


def test_overlapping_tiles_detect_small_changes_without_global_dilution():
    a = Image.new("RGB", (1280, 800), "white")
    pixels = np.asarray(a).copy()
    pixels[30:35, 30:35] = 0
    result = compare_screens(a, Image.fromarray(pixels))
    assert result.changed and result.changed_ratio < 0.0005 and result.tile_changed_ratio >= 0.015
    assert not compare_screens(a, a).changed
    assert not compare_screens(a, Image.fromarray(pixels), [(29, 29, 36, 36)]).changed
    assert compare_screens(a, Image.fromarray(pixels), [(-100, -100, -20, -20)]).changed
    assert compare_screens(a, a, [(0, 0, 1280, 800)]).changed


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["timeout", "cancel", "stop", "close", "cancel_during_spawn"])
async def test_hung_native_worker_is_killed_and_reaped_without_blocking_loop(tmp_path, monkeypatch, action):
    original = asyncio.create_subprocess_exec
    processes = []
    created = asyncio.Event()

    async def spawn(*_args, **_kwargs):
        process = await original(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        created.set()
        if action == "cancel_during_spawn":
            await asyncio.sleep(0.05)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    stop = asyncio.Event()
    service = ScreenCapture(timeout=0.15 if action == "timeout" else 10)
    await service.start()
    pending = asyncio.create_task(service.capture(tmp_path, stop_event=stop))
    await asyncio.wait_for(created.wait(), timeout=2)
    # This timer remains runnable while native capture is hung in another process.
    await asyncio.wait_for(asyncio.sleep(0.01), timeout=1)
    with pytest.raises(RuntimeError, match="already running"):
        await service.capture(tmp_path)
    assert len(processes) == 1
    if action.startswith("cancel"):
        pending.cancel()
        await asyncio.sleep(0)
        pending.cancel()
    elif action == "stop":
        stop.set()
    elif action == "close":
        await service.close()
    error = TimeoutError if action == "timeout" else RuntimeError if action == "close" else asyncio.CancelledError
    with pytest.raises(error):
        await asyncio.wait_for(pending, timeout=2)
    assert processes[0].returncode is not None and service._process is None and service._launch is None
    await service.close()


@pytest.mark.asyncio
async def test_capture_slot_is_bounded_and_closing_revokes_it():
    service = ScreenCapture()
    await service.start()
    async with service.operation():
        with pytest.raises(RuntimeError, match="already running"):
            async with service.operation():
                pytest.fail("overlapping capture was admitted")
    await service.close()
    with pytest.raises(RuntimeError, match="closed"):
        async with service.operation():
            pytest.fail("closed capture was admitted")
