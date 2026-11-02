"""Evaluate a BEAM Home candidate in an independent test workspace."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.home_eval import main, run as run_evaluation  # noqa: E402  # pylint: disable=wrong-import-position

DEFAULT_CONFIG = Path(__file__).with_name("config_home_eval.yaml")


async def run(config_path=DEFAULT_CONFIG, **kwargs):
    """Score the selected BEAM commit without changing its optimization workspace."""
    return await run_evaluation(config_path, expected_benchmark="beam", **kwargs)


if __name__ == "__main__":
    main("beam", DEFAULT_CONFIG)
