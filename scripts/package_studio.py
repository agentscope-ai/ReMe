"""Prepare the optional ReMe Studio Python distribution."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
WEBSITE_DIR = REPOSITORY_DIR / "website"
PACKAGE_DIR = REPOSITORY_DIR / "packages" / "reme_ai_studio"
STATIC_DIR = PACKAGE_DIR / "src" / "reme_ai_studio" / "static"
LICENSE_FILE = REPOSITORY_DIR / "LICENSE"
STATIC_GITIGNORE = "*\n!.gitignore\n"

_RAW_WEBSITE_URL = "https://raw.githubusercontent.com/agentscope-ai/ReMe/main/website"
_REPOSITORY_URL = "https://github.com/agentscope-ai/ReMe"


def build_readme() -> str:
    """Compose the PyPI description from the English and Chinese Studio docs."""
    english = (WEBSITE_DIR / "README.md").read_text(encoding="utf-8")
    chinese = (WEBSITE_DIR / "README_ZH.md").read_text(encoding="utf-8")
    english = english.replace("English | [简体中文](./README_ZH.md)", "English | [简体中文](#简体中文)")
    chinese = chinese.replace(
        "# ReMe Studio\n\n[English](./README.md) | 简体中文",
        "# 简体中文\n\n[English](#reme-studio) | 简体中文",
    )
    for relative, absolute in {
        "./public/og.jpg": f"{_RAW_WEBSITE_URL}/public/og.jpg",
        "../README.md": f"{_REPOSITORY_URL}#readme",
        "../README_ZH.md": f"{_REPOSITORY_URL}/blob/main/README_ZH.md",
    }.items():
        english = english.replace(relative, absolute)
        chinese = chinese.replace(relative, absolute)
    return f"{english.rstrip()}\n\n---\n\n{chinese.rstrip()}\n"


def prepare_package(*, copy_static: bool = True) -> None:
    """Generate package metadata files and optionally stage the static build."""
    (PACKAGE_DIR / "README.md").write_text(build_readme(), encoding="utf-8")
    shutil.copyfile(LICENSE_FILE, PACKAGE_DIR / "LICENSE")
    if not copy_static:
        return
    source = WEBSITE_DIR / "dist-static"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme-only", action="store_true", help="Generate only the PyPI README")
    args = parser.parse_args()
    prepare_package(copy_static=not args.readme_only)


if __name__ == "__main__":
    main()
