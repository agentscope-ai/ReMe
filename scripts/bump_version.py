"""Validate or update the ReMe and ReMe Studio release version."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPOSITORY_DIR = Path(__file__).resolve().parents[1]

_VERSION_PATTERN = re.compile(r'(?m)^__version__ = "(?P<version>[^"]+)"$')
_STUDIO_VERSION_PATTERN = re.compile(r'(?m)^version = "(?P<version>[^"]+)"$')


def _paths(repository: Path) -> tuple[Path, Path, Path]:
    return (
        repository / "reme" / "__init__.py",
        repository / "pyproject.toml",
        repository / "packages" / "reme_ai_studio" / "pyproject.toml",
    )


def _canonical_version(value: str) -> str:
    """Return one canonical PEP 440 version or explain how to correct it."""
    try:
        canonical = str(Version(value))
    except InvalidVersion as error:
        raise ValueError(
            f"Invalid PEP 440 version {value!r}. Use a version such as 0.4.1.8, then rerun this command.",
        ) from error
    if canonical != value:
        raise ValueError(f"Version {value!r} is not canonical PEP 440; use {canonical!r} instead.")
    return canonical


def _match_version(text: str, pattern: re.Pattern[str], source: Path) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one version declaration in {source}, found {len(matches)}.")
    return matches[0].group("version")


def _replace_version(text: str, pattern: re.Pattern[str], version: str, source: Path) -> str:
    _match_version(text, pattern, source)
    return pattern.sub(lambda match: match.group(0).replace(match.group("version"), version), text)


def _write_atomic(path: Path, content: str) -> None:
    """Replace one text file without exposing a partially written file."""
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_name = temporary.name
        os.chmod(temporary_name, path.stat().st_mode)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_version(repository: Path = REPOSITORY_DIR) -> str:
    """Read and validate the main distribution version."""
    version_file, _, _ = _paths(repository)
    version = _match_version(version_file.read_text(encoding="utf-8"), _VERSION_PATTERN, version_file)
    return _canonical_version(version)


def check_versions(repository: Path = REPOSITORY_DIR, expected_version: str | None = None) -> str:
    """Validate both distributions and their exact optional-dependency pins."""
    version_file, main_config_file, studio_config_file = _paths(repository)
    version = read_version(repository)
    main_config = tomllib.loads(main_config_file.read_text(encoding="utf-8"))
    studio_config = tomllib.loads(studio_config_file.read_text(encoding="utf-8"))
    dependency = f"reme-ai-studio=={version}"
    optional_dependencies = main_config["project"]["optional-dependencies"]
    problems: list[str] = []

    studio_version = studio_config["project"].get("version")
    if studio_version != version:
        problems.append(f"{studio_config_file}: project.version is {studio_version!r}; expected {version!r}")
    if optional_dependencies.get("web") != [dependency]:
        problems.append(f"{main_config_file}: web must be exactly [{dependency!r}]")
    if optional_dependencies.get("core", []).count(dependency) != 1:
        problems.append(f"{main_config_file}: core must contain {dependency!r} exactly once")

    expected = None
    if expected_version is not None:
        expected = _canonical_version(expected_version.removeprefix("v"))
        if version != expected:
            problems.append(f"{version_file}: package version is {version!r}; release expects {expected!r}")

    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(
            "Version metadata is inconsistent:\n"
            f"{details}\n"
            f"Fix the listed values, or run `python scripts/bump_version.py {expected or version}` "
            "after restoring a consistent current version.",
        )
    return version


def bump_version(version: str, repository: Path = REPOSITORY_DIR) -> str:
    """Update every release version, rolling back if any write or validation fails."""
    version = _canonical_version(version)
    current_version = check_versions(repository)
    version_file, main_config_file, studio_config_file = _paths(repository)
    paths = (version_file, main_config_file, studio_config_file)
    original = {path: path.read_text(encoding="utf-8") for path in paths}
    current_dependency = f"reme-ai-studio=={current_version}"

    main_text = original[main_config_file]
    if main_text.count(current_dependency) != 2:
        raise ValueError(f"Expected exactly two {current_dependency!r} pins in {main_config_file}.")
    updated = {
        version_file: _replace_version(original[version_file], _VERSION_PATTERN, version, version_file),
        main_config_file: main_text.replace(current_dependency, f"reme-ai-studio=={version}"),
        studio_config_file: _replace_version(
            original[studio_config_file],
            _STUDIO_VERSION_PATTERN,
            version,
            studio_config_file,
        ),
    }

    written: list[Path] = []
    try:
        for path in paths:
            _write_atomic(path, updated[path])
            written.append(path)
        check_versions(repository, version)
    except BaseException as error:
        rollback_errors: list[str] = []
        for path in reversed(written):
            try:
                _write_atomic(path, original[path])
            except OSError as rollback_error:
                rollback_errors.append(f"{path}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "Version update failed and rollback was incomplete. Restore these files from Git:\n- "
                + "\n- ".join(rollback_errors),
            ) from error
        raise RuntimeError(
            "Version update failed; all changed files were restored. Fix the error and retry.",
        ) from error
    return current_version


def main() -> None:
    """Run the version validation or update command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="New version for both distributions, for example 0.4.1.8")
    parser.add_argument("--check", action="store_true", help="Validate versions without changing files")
    parser.add_argument("--expected-version", help="Also require this release/tag version; a leading v is accepted")
    args = parser.parse_args()
    if args.check:
        if args.version:
            parser.error("version cannot be used with --check; use --expected-version")
        version = check_versions(expected_version=args.expected_version)
        print(f"Release versions are consistent: {version}")
        return
    if args.expected_version:
        parser.error("--expected-version requires --check")
    if not args.version:
        parser.error("provide a version to update, or use --check")
    previous_version = bump_version(args.version)
    print(f"Updated ReMe and ReMe Studio from {previous_version} to {args.version}")


if __name__ == "__main__":
    main()
