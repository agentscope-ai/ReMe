"""Capture every display separately without merging or rescaling images."""

from datetime import datetime
from pathlib import Path

from .base_connector import BaseConnectorStep
from ....components import R
from ....schema import ConnectorResult


@R.register("screen_capture_step")
class ScreenCaptureStep(BaseConnectorStep):
    """Collect one PNG per display; a failed display aborts the whole package."""

    source = "screen_capture"

    def collect(self, directory: Path, created_at: datetime) -> ConnectorResult:
        try:
            import mss
            from mss.tools import to_png
        except ImportError as exc:
            raise RuntimeError("Screen capture requires mss; install reme-ai[core]") from exc

        paths = []
        with mss.mss() as capture:
            monitors = capture.monitors[1:]
            if not monitors:
                raise RuntimeError("No displays available for screen capture")
            for display_id, monitor in enumerate(monitors, 1):
                image = capture.grab(monitor)
                filename = f"{created_at:%Y-%m-%d_%H-%M-%S-%f}_screen_{display_id}.png"
                to_png(image.rgb, image.size, output=str(directory / filename))
                paths.append(filename)
        return ConnectorResult(
            context=(
                f"Captured {len(paths)} displays at {created_at.isoformat()}, one original-size PNG per display. "
                "Displays were captured sequentially. screen_1, screen_2, etc. are display numbers for this collection. "
                "These images show the desktop at capture time, not the current live desktop. "
                "Read the listed images as needed to understand their content."
            ),
            paths=paths,
        )
