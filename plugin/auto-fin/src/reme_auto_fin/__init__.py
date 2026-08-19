"""Auto Fin news research workflow."""

from pathlib import Path

from .data import AutoFinDataStep
from .merge import AutoFinMergeStep
from .plugin import plugin
from .schema import AutoFinReportOutput, AutoFinTopicOutput
from .topic import AutoFinTopicStep

CONFIG_PATH = Path(__file__).with_name("config.yaml")

__all__ = [
    "AutoFinReportOutput",
    "AutoFinDataStep",
    "AutoFinMergeStep",
    "AutoFinTopicOutput",
    "AutoFinTopicStep",
    "CONFIG_PATH",
    "plugin",
]
