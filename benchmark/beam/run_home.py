"""Prepare the beam Home workspace, dataset and candidate code; do not evaluate."""

from pathlib import Path
import sys

# Support direct execution from any working directory as well as module imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.home_runner import main, prepare  # noqa: E402  # pylint: disable=wrong-import-position

DEFAULT_CONFIG = Path(__file__).with_name("config_home.yaml")


async def run(config_path: str | Path = DEFAULT_CONFIG, *, workspace_dir: str | None = None) -> dict:
    """Prepare through the same Home Job as the CLI, then close the application."""
    return await prepare(config_path, workspace_dir=workspace_dir)


if __name__ == "__main__":
    main(DEFAULT_CONFIG)
