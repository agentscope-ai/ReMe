"""Re-score current-workflow results without changing their recorded outcomes."""

import argparse
from pathlib import Path

from benchmark.proactive.evaluate import report
from benchmark.proactive.runtime import save


def analyze(root: Path) -> dict:
    """Re-score a recorded run and save its analysis separately."""
    result = report(root, rescore=True)
    save(root / "analysis" / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root.expanduser().resolve())
