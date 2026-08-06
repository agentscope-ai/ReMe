"""Launch Meta-ReMe for BEAM using one YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Forward the selected YAML configuration to Meta-ReMe."""

    parser = argparse.ArgumentParser(description="Run Meta-ReMe for BEAM")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_meta_reme.yaml")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(PROJECT_ROOT / "meta-reme/run.py"),
        "--config",
        str(args.config.resolve()),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
