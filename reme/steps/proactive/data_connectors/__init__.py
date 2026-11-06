"""Collect external observations into workspace resource packages."""

from .base_connector import BaseConnectorStep
from .screen_capture import ScreenCaptureStep

__all__ = ["BaseConnectorStep", "ScreenCaptureStep"]
