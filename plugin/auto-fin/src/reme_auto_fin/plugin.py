"""Auto Fin plugin declaration."""

from pathlib import Path

import yaml

from reme.plugin import Backend, Plugin

from .data import AutoFinDataStep
from .merge import AutoFinMergeStep
from .topic import AutoFinTopicStep


def _default_config() -> dict:
    path = Path(__file__).with_name("defaults.yaml")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Auto Fin defaults must be a mapping: {path}")
    return config


plugin = Plugin(
    name="auto-fin",
    backends=(
        Backend("auto_fin_data_step", AutoFinDataStep),
        Backend("auto_fin_topic_step", AutoFinTopicStep),
        Backend("auto_fin_merge_step", AutoFinMergeStep),
    ),
    config=_default_config(),
)
