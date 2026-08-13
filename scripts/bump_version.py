"""Update the versions of the ReMe and ReMe Studio distributions."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import tomllib
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]

_VERSION_PATTERN = re.compile(r'(?m)^__version__ = "(?P<version>[^"]+)"$')
_STUDIO_PROJECT_VERSION_PATTERN = re.compile(r'(?m)^version = "(?P<version>[^"]+)"$')
_SAFE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$")


def _replace_once(text: str, pattern: re.Pattern[str], replacement: str, source: Path) -> str:
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"Expected exactly one version declaration in {source}, found {count}")
    return updated


def _write_atomic(path: Path, content: str) -> None:
    """Replace one text file without exposing a partially written file."""
    mode = path.stat().st_mode
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_name = temporary.name
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_version(repository: Path = REPOSITORY_DIR) -> str:
    """Read the main distribution version from the repository source."""
    version_file = repository / "reme" / "__init__.py"
    match = _VERSION_PATTERN.search(version_file.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Version declaration is missing from {version_file}")
    return match.group("version")


def _load_sources(repository: Path) -> tuple[dict[Path, str], str]:
    """Load all versioned files and reject an inconsistent starting state."""
    version_file = repository / "reme" / "__init__.py"
    main_config_file = repository / "pyproject.toml"
    studio_config_file = repository / "packages" / "reme_ai_studio" / "pyproject.toml"
    sources = {path: path.read_text(encoding="utf-8") for path in (version_file, main_config_file, studio_config_file)}
    current_version = read_version(repository)
    current_dependency = f"reme-ai-studio=={current_version}"
    main_config = tomllib.loads(sources[main_config_file])
    studio_config = tomllib.loads(sources[studio_config_file])
    optional_dependencies = main_config["project"]["optional-dependencies"]
    if optional_dependencies.get("web") != [current_dependency]:
        raise ValueError(f"The web extra in {main_config_file} is not pinned to {current_dependency}")
    if optional_dependencies.get("core", []).count(current_dependency) != 1:
        raise ValueError(f"The core extra in {main_config_file} is not pinned once to {current_dependency}")
    if studio_config["project"].get("version") != current_version:
        raise ValueError(f"The Studio package in {studio_config_file} does not match {current_version}")
    return sources, current_version


def bump_version(version: str, repository: Path = REPOSITORY_DIR) -> str:
    """Validate the current package versions, then update every release version."""
    if not _SAFE_VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Invalid version: {version!r}")

    version_file = repository / "reme" / "__init__.py"
    main_config_file = repository / "pyproject.toml"
    studio_config_file = repository / "packages" / "reme_ai_studio" / "pyproject.toml"
    sources, current_version = _load_sources(repository)
    current_dependency = f"reme-ai-studio=={current_version}"

    next_dependency = f"reme-ai-studio=={version}"
    updated_version_text = _replace_once(
        sources[version_file],
        _VERSION_PATTERN,
        f'__version__ = "{version}"',
        version_file,
    )
    updated_main_config_text, dependency_count = sources[main_config_file].replace(
        current_dependency,
        next_dependency,
    ), sources[main_config_file].count(current_dependency)
    if dependency_count != 2:
        raise ValueError(f"Expected exactly two Studio pins in {main_config_file}, found {dependency_count}")
    updated_studio_config_text = _replace_once(
        sources[studio_config_file],
        _STUDIO_PROJECT_VERSION_PATTERN,
        f'version = "{version}"',
        studio_config_file,
    )

    for path, content in (
        (version_file, updated_version_text),
        (main_config_file, updated_main_config_text),
        (studio_config_file, updated_studio_config_text),
    ):
        _write_atomic(path, content)
    return current_version


def main() -> None:
    """Run the version update command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="New version for both distributions, for example 0.4.1.8")
    args = parser.parse_args()
    previous_version = bump_version(args.version)
    print(f"Updated ReMe and ReMe Studio from {previous_version} to {args.version}")


if __name__ == "__main__":
    main()
