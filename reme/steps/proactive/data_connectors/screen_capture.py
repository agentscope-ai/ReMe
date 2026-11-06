"""Capture every display separately without merging or rescaling images."""

from pathlib import Path

from .base_connector import BaseConnectorStep
from ....components import R


@R.register("screen_capture_step")
class ScreenCaptureStep(BaseConnectorStep):
    """Collect one PNG per display; a failed display aborts the whole package."""

    source = "screen"
    description = (
        "Screenshots of all displays from one collection. Read every listed image together to understand "
        "the user's activity at the recorded time. Display IDs are enumeration numbers for this collection. "
        "These are historical observations, not the current live desktop."
    )

    def collect(self, directory: Path) -> list[dict]:
        try:
            import mss
            from mss.tools import to_png
        except ImportError as exc:
            raise RuntimeError("Screen capture requires mss; install reme-ai[core]") from exc

        files = []
        with mss.mss() as capture:
            monitors = capture.monitors[1:]
            if not monitors:
                raise RuntimeError("No displays available for screen capture")
            for display_id, monitor in enumerate(monitors, 1):
                image = capture.grab(monitor)
                filename = f"{display_id}.png"
                to_png(image.rgb, image.size, output=str(directory / filename))
                files.append({"path": filename})
        return files
