"""Evaluate a LongMemEval Home candidate on an independent test dataset."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.home_eval import main, run as run_evaluation  # noqa: E402  # pylint: disable=wrong-import-position

DEFAULT_CONFIG = Path(__file__).with_name("config_home_eval.yaml")


async def run(config_path=DEFAULT_CONFIG, **kwargs):
    """Score the selected LongMemEval commit in its own Home workspace."""
    return await run_evaluation(config_path, expected_benchmark="longmemeval", **kwargs)


if __name__ == "__main__":
    main("longmemeval", DEFAULT_CONFIG)
