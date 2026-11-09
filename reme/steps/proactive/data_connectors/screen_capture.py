"""Compare original per-display captures before publishing changed image paths."""

from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from .base_connector import BaseConnectorStep
from ...base_step import Ref
from ....components import R, ScreenCapture
from ....enumeration import ComponentEnum
from ....schema import ConnectorResult
from ....schema.screen_capture import CapturedDisplay
from ....utils.screen_compare import compare_screens
from ...file_io._path import _check_path_permission

_PNG_METADATA = "reme_screen_capture"


@R.register("screen_capture_step")
class ScreenCaptureStep(BaseConnectorStep):
    """Return context + changed PNG paths; successful task summaries own their baselines."""

    source = "screen_capture"
    screen_capture: ScreenCapture = Ref(ScreenCapture, ComponentEnum.SCREEN_CAPTURE, optional=True)
    _service: ScreenCapture
    _capture_time: datetime
    _raw_directory: Path
    _raw_frames: list[dict]
    _next_baseline: list[CapturedDisplay]

    async def execute(self):
        self.resource_target()  # Validate containment before native collection.
        async with AsyncExitStack() as stack:
            service = self.screen_capture
            if service is None:
                if self.app_context is not None:
                    raise RuntimeError("Configure a screen_capture component before using screen_capture_step")
                service = await stack.enter_async_context(ScreenCapture())
            self._service = service
            self._capture_time = self.collection_time()
            async with service.operation():
                with TemporaryDirectory(prefix=".reme-screen-", dir=self.workspace_path.resolve()) as temporary:
                    self._raw_directory = Path(temporary)
                    self._raw_frames = await service.capture(self._raw_directory, stop_event=self.context.stop_event)
                    response = await super().execute()
                    response.metadata["screen_capture"] = [
                        frame.model_dump(mode="json") for frame in self._next_baseline
                    ]
                    return response

    def _manual_baseline(self, token: str, target: Path):
        from PIL import Image

        for path in sorted(target.glob(f"*_display_{token}.png"), reverse=True):
            if path.resolve() != path.absolute():
                continue
            try:
                with Image.open(path) as image:
                    frame = CapturedDisplay.model_validate_json(image.info.get(_PNG_METADATA, ""))
                if frame.scope == "manual" and frame.path == self.to_workspace_relative(path):
                    return frame
            except (OSError, ValueError):
                continue
        return None

    def _different(self, previous, current, image) -> bool:
        from PIL import Image

        if previous is None or not previous.stable_id or not current.stable_id:
            return True
        if (previous.display_id, previous.pixel_size, previous.bounds[2:], previous.rotation) != (
            current.display_id,
            current.pixel_size,
            current.bounds[2:],
            current.rotation,
        ):
            return True
        path = self.workspace_path / previous.path
        if (
            path.resolve() != path.absolute()
            or not path.resolve().is_relative_to(self.workspace_path.resolve())
            or not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths"))
        ):
            return True
        try:
            with Image.open(path) as old:
                if old.size != previous.pixel_size:
                    return True
                return compare_screens(
                    old,
                    image,
                    previous.ignored_regions + current.ignored_regions,
                    **self._service.comparison,
                ).changed
        except (OSError, ValueError):
            return True

    def collect(self, directory: Path, created_at: datetime) -> ConnectorResult:
        from PIL import Image, PngImagePlugin

        created_at = self._capture_time
        target = self.resource_target()
        scope = self.context.get("_screen_scope", self.kwargs.get("scope", "manual"))
        baseline = self.context.get("_screen_baseline", self.kwargs.get("baseline"))
        previous = {
            frame.display_id: frame
            for frame in (CapturedDisplay.model_validate(value) for value in baseline or [])
            if frame.scope == scope
        }
        paths, changed, self._next_baseline = [], [], []
        identities = set()
        for raw in self._raw_frames:
            self._check_stopped(self._cancelled)
            current = CapturedDisplay(**raw, scope=scope, captured_at=created_at)
            if Path(current.path).name != current.path or current.display_id in identities:
                raise ValueError("Invalid or duplicate captured display metadata")
            identities.add(current.display_id)
            source = self._raw_directory / current.path
            token = Path(current.path).stem.removeprefix("display_")
            old = previous.get(current.display_id)
            if baseline is None and scope == "manual":
                old = self._manual_baseline(token, target)
            with Image.open(source) as image:
                if image.size != current.pixel_size:
                    raise ValueError("Captured display dimensions do not match its pixels")
                if not self._different(old, current, image):
                    self._next_baseline.append(old)
                    continue
                filename = f"{created_at:%Y-%m-%d_%H-%M-%S-%f}_display_{token}.png"
                current = current.model_copy(update={"path": self.to_workspace_relative(target / filename)})
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text(_PNG_METADATA, current.model_dump_json())
                image.save(directory / filename, pnginfo=metadata)
            paths.append(filename)
            changed.append(current.display_id)
            self._next_baseline.append(current)
        if not paths:
            context = "所有显示器均与上次处理的截图基本一致，没有明显内容变化。"
        else:
            context = (
                f"Captured {len(self._raw_frames)} displays at {created_at.isoformat()}; "
                f"{len(paths)} changed displays: {', '.join(changed)}. "
                "Only changed original-size PNGs are listed. Displays were captured sequentially, without the pointer. "
                "Read these images for the current query; they are snapshots, not the live desktop."
            )
        return ConnectorResult(context=context, paths=paths)
