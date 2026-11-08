"""One lifecycle-owned native capture process at a time, with a hard deadline."""

import asyncio
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import sys

from .base_component import BaseComponent
from .component_registry import R
from ..enumeration import ComponentEnum
from ..utils.async_utils import finish_cleanup


@R.register("local")
class ScreenCapture(BaseComponent):
    """Native screenshot APIs run outside the service process, including on macOS."""

    component_type = ComponentEnum.SCREEN_CAPTURE

    def __init__(
        self,
        timeout: float = 90,
        long_edge: int = 1280,
        pixel_delta: int = 20,
        changed_ratio: float = 0.0005,
        tile_size: int = 32,
        tile_stride: int = 16,
        tile_changed_ratio: float = 0.015,
        ignore_clock: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Screen capture timeout must be positive and finite")
        for name, value, maximum in (
            ("long_edge", long_edge, 4096),
            ("pixel_delta", pixel_delta, 254),
            ("tile_size", tile_size, 256),
            ("tile_stride", tile_stride, tile_size),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Invalid screen comparison {name}")
        for value in (changed_ratio, tile_changed_ratio):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Screen change ratios must be in (0, 1]")
        if type(ignore_clock) is not bool:
            raise ValueError("ignore_clock must be a boolean")
        self.timeout = timeout
        self.ignore_clock = ignore_clock
        self.comparison = {
            "long_edge": long_edge,
            "pixel_delta": pixel_delta,
            "changed_ratio": changed_ratio,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "tile_changed_ratio": tile_changed_ratio,
        }
        self._capture_lock = asyncio.Lock()
        self._capture_task: asyncio.Task | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._launch: asyncio.Task | None = None
        self._closed = False

    @asynccontextmanager
    async def operation(self):
        """Reject overlapping requests instead of building an unbounded capture queue."""
        if self._closed:
            raise RuntimeError("Screen capture component is closed")
        if self._capture_lock.locked():
            raise RuntimeError("Screen capture is already running")
        async with self._capture_lock:
            self._capture_task = asyncio.current_task()
            try:
                yield
            finally:
                self._capture_task = None

    async def capture(self, directory: Path, *, stop_event: asyncio.Event | None = None) -> list[dict]:
        """Kill and reap native work on timeout, cancellation, or component shutdown."""
        if self._capture_task is asyncio.current_task():
            return await self._capture(directory, stop_event=stop_event)
        async with self.operation():
            return await self._capture(directory, stop_event=stop_event)

    async def _capture(self, directory: Path, *, stop_event: asyncio.Event | None = None) -> list[dict]:
        worker = Path(__file__).parents[1] / "utils" / "screen_capture_worker.py"
        if stop_event is not None and stop_event.is_set():
            raise asyncio.CancelledError()
        launch = self._launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                str(worker),
                str(directory),
                "1" if self.ignore_clock else "0",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
        )
        try:
            process = await asyncio.shield(launch)
        except BaseException:
            await finish_cleanup(self._cancel_launch(launch))
            raise
        self._process = process
        output = asyncio.create_task(process.communicate())
        stopped = asyncio.create_task(stop_event.wait()) if stop_event is not None else None
        try:
            if self._closed:
                raise RuntimeError("Screen capture component is closed")
            completed, _pending = await asyncio.wait(
                {output, stopped} if stopped is not None else {output},
                timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopped is not None and stopped in completed:
                raise asyncio.CancelledError()
            if output not in completed:
                raise TimeoutError(f"Screen capture exceeded {self.timeout:g} seconds")
            stdout, stderr = output.result()
            if process.returncode:
                raise RuntimeError(f"Screen capture failed: {stderr.decode(errors='replace')[-2000:]}")
            if len(stdout) > 1024 * 1024:
                raise ValueError("Screen capture returned excessive metadata")
            frames = json.loads(stdout)
            if not isinstance(frames, list) or not 1 <= len(frames) <= 32:
                raise RuntimeError("No displays available for screen capture")
            return frames
        finally:
            await finish_cleanup(self._finish(process, output, stopped))

    async def _cancel_launch(self, launch):
        try:
            process = await launch
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.communicate()
        finally:
            self._launch = None

    async def _finish(self, process, output, stopped):
        try:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await output
        finally:
            if stopped is not None:
                stopped.cancel()
                await asyncio.gather(stopped, return_exceptions=True)
            self._process = self._launch = None

    async def _start(self):
        self._closed = False

    async def _close(self):
        self._closed = True
        process = self._process
        if process is None and self._launch is not None:
            process = await asyncio.shield(self._launch)
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
