"""Capture changed display groups without merging or rescaling images."""

from hashlib import file_digest
import os
from pathlib import Path
from stat import S_ISREG

from .base_connector import BaseConnectorStep
from ...file_io import check_path_permission, get_path_lock
from ....components import R
from ....components.agent_runtime._internal.files import AgentFiles
from ....schema import ResourceMetadata
from ....schema.resource_metadata import ResourceFile


@R.register("screen_capture_step")
class ScreenCaptureStep(BaseConnectorStep):
    """Collect one PNG per display; a failed display aborts the whole package."""

    source = "screenshot"
    description = (
        "Screenshots of all displays from one collection. Read every listed image together to understand "
        "the user's activity at the recorded time. Display IDs are enumeration numbers for this collection. "
        "These are historical observations, not the current live desktop."
    )

    async def execute(self):
        target, _ = self.resource_target()
        async with await get_path_lock(target.parent.parent):
            return await super().execute()

    def _signature(self, folder, entries):
        files = AgentFiles(self.workspace_path)
        result = []
        for entry in sorted(entries, key=lambda item: item.path):
            path = files.checked(folder / entry.path)
            if not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to read screenshot history")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as handle:
                if not S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError("Screenshot payload must be a regular file")
                result.append((entry.model_dump(exclude_none=True), file_digest(handle, "sha256").hexdigest()))
        return result

    def _unchanged(self, directory, entries):
        files = AgentFiles(self.workspace_path)
        latest = None
        for manifest in directory.parent.parent.glob("*/*/metadata.json"):
            files.checked(manifest)
            if not check_path_permission(self.workspace_path, manifest, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to read screenshot history")
            metadata = ResourceMetadata.model_validate_json(files.read(manifest))
            if metadata.source != self.source:
                raise ValueError(f"Unexpected screenshot source: {metadata.source}")
            if latest is None or metadata.created_at > latest[1].created_at:
                latest = manifest, metadata
        return latest is not None and self._signature(directory, entries) == self._signature(
            latest[0].parent,
            latest[1].files,
        )

    def collect(self, directory: Path) -> list[dict]:
        try:
            import mss
            from mss.tools import to_png
        except ImportError as exc:
            raise RuntimeError("Screen capture requires mss; install reme-ai[core]") from exc

        files = []
        with mss.MSS() as capture:
            monitors = capture.monitors[1:]
            if not monitors:
                raise RuntimeError("No displays available for screen capture")
            for display_id, monitor in enumerate(monitors, 1):
                image = capture.grab(monitor)
                filename = f"{display_id}.png"
                to_png(image.rgb, image.size, output=str(directory / filename))
                files.append({"path": filename, "monitor": dict(monitor)})
        if self._unchanged(directory, [ResourceFile.model_validate(entry) for entry in files]):
            return []
        return files
