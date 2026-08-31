"""Stage the ReMe Studio static build for its Python distribution."""

from __future__ import annotations

import shutil
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
STUDIO_DIR = REPOSITORY_DIR / "reme_studio"
STATIC_DIR = STUDIO_DIR / "src" / "reme_studio" / "static"
LICENSE_FILE = REPOSITORY_DIR / "LICENSE"
STATIC_GITIGNORE = "*\n!.gitignore\n"


def prepare_package() -> None:
    """Copy generated assets into the importable Python package."""
    shutil.copyfile(LICENSE_FILE, STUDIO_DIR / "LICENSE")
    source = STUDIO_DIR / "dist-static"
    if not (source / "index.html").is_file():
        raise FileNotFoundError(f"Studio static build is unavailable: {source}")
    try:
        shutil.rmtree(STATIC_DIR, ignore_errors=True)
        shutil.copytree(source, STATIC_DIR)
    finally:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        (STATIC_DIR / ".gitignore").write_text(STATIC_GITIGNORE, encoding="utf-8")


def main() -> None:
    """Run the package preparation command."""
    prepare_package()


if __name__ == "__main__":
    main()
