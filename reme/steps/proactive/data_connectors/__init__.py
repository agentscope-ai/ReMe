"""Collect external observations into workspace resource packages."""

from .base_connector import BaseConnectorStep
from .screen_capture import ScreenCaptureStep
from .event import ResourceEventStep

__all__ = ["BaseConnectorStep", "ScreenCaptureStep", "ResourceEventStep"]
