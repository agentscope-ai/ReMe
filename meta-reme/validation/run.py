"""CLI entry point for validating prepared Meta-ReMe cases."""

# Import roots must be installed before importing the hyphenated meta-reme tree.
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from pathlib import Path
import sys

META_REME_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = META_REME_ROOT.parent
for import_root in (META_REME_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from validation.evaluator import run_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse validation inputs without consulting mutable workspace state."""

    parser = argparse.ArgumentParser(description="Construct memory and validate immutable ReMe code in sandboxes")
    parser.add_argument("--workspace", type=Path, required=True, help="Prepared Meta-ReMe workspace")
    parser.add_argument(
        "--case-id",
        action="append",
        required=True,
        dest="case_ids",
        help="Prepared case ID; repeat this option to validate multiple cases",
    )
    parser.add_argument("--code-id", required=True, help="Path-safe local Git branch name identifying the code")
    parser.add_argument("--concurrency", type=int, required=True, help="Maximum concurrent case sandboxes")
    parser.add_argument("--validation-id", help="Optional non-overwriting validation run ID")
    return parser.parse_args()


def main() -> None:
    """Run validation and report the immutable result directory."""

    args = parse_args()
    try:
        output = run_validation(
            args.workspace,
            args.case_ids,
            args.code_id,
            args.concurrency,
            validation_id=args.validation_id,
        )
    except Exception as exc:
        raise SystemExit(f"validation failed: {exc}") from exc
    print(f"Validation results: {output}")


if __name__ == "__main__":
    main()
