"""Validate or update the ReMe release version."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPOSITORY_DIR = Path(__file__).resolve().parents[1]

_SHARED_VERSION = re.compile(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}")
_STUDIO_MANIFESTS = ("desktop/package.json", "web/package.json", "web/python/pyproject.toml", "package-lock.json")

_VERSION_PATTERN = re.compile(r'(?m)^__version__ = "(?P<version>[^"]+)"$')


def _version_file(repository: Path) -> Path:
    return repository / "reme" / "_version.py"


def _canonical_version(value: str) -> str:
    """Return one canonical PEP 440 version or explain how to correct it."""
    try:
        canonical = str(Version(value))
    except InvalidVersion as error:
        raise ValueError(
            f"Invalid PEP 440 version {value!r}. Use a version such as 0.4.3, then rerun this command.",
        ) from error
    if canonical != value:
        raise ValueError(f"Version {value!r} is not canonical PEP 440; use {canonical!r} instead.")
    return canonical


def _match_version(text: str, source: Path) -> str:
    matches = list(_VERSION_PATTERN.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one version declaration in {source}, found {len(matches)}.")
    return matches[0].group("version")


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
    """Read and validate the ReMe distribution version."""
    version_file = _version_file(repository)
    version = _match_version(version_file.read_text(encoding="utf-8"), version_file)
    try:
        return _canonical_version(version)
    except ValueError as error:
        raise ValueError(f"{version_file}: {error} Fix the __version__ declaration and retry.") from error


def _studio_versions(repository: Path, version: str) -> dict[Path, str]:
    studio = repository / "reme_studio"
    updates = {}
    for name in _STUDIO_MANIFESTS:
        path = studio / name
        original = path.read_text(encoding="utf-8")
        if path.suffix == ".toml":
            updated, count = re.subn(r'(?m)^version = "[^"]+"$', f'version = "{version}"', original)
            if count != 1:
                raise ValueError(f"Missing Studio distribution version in {path}.")
        else:
            manifest = json.loads(original)
            entries = (
                [manifest["packages"][key] for key in ("desktop", "web")]
                if path.name == "package-lock.json"
                else [manifest]
            )
            if all(entry["version"] == version for entry in entries):
                continue
            for entry in entries:
                entry["version"] = version
            updated = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        if updated != original:
            updates[path] = updated
    path = repository / "pyproject.toml"
    original = path.read_text(encoding="utf-8")
    updated, count = re.subn(r'"reme_studio==[^"]+"', f'"reme_studio=={version}"', original)
    if count != 1:
        raise ValueError(f"Expected one Studio requirement in {path}.")
    if updated != original:
        updates[path] = updated
    return updates


def check_versions(repository: Path = REPOSITORY_DIR, expected_version: str | None = None) -> str:
    """Validate the ReMe version and an optional release tag."""
    version = read_version(repository)
    if not _SHARED_VERSION.fullmatch(version):
        raise ValueError(f"Use a shared Python and npm version in X.Y.Z format, got {version!r}.")
    expected = _canonical_version(expected_version.removeprefix("v")) if expected_version else version
    if version != expected:
        version_file = _version_file(repository)
        raise ValueError(f"{version_file}: package version is {version!r}; release expects {expected!r}")
    if outdated := _studio_versions(repository, version):
        raise ValueError("Studio manifests differ from reme/_version.py: " + ", ".join(str(path) for path in outdated))
    return version


def bump_version(version: str, repository: Path = REPOSITORY_DIR) -> str:
    """Update the ReMe and Studio distribution versions together."""
    version = _canonical_version(version)
    if not _SHARED_VERSION.fullmatch(version):
        raise ValueError(f"Use a shared Python and npm version in X.Y.Z format, got {version!r}.")
    previous_version = read_version(repository)
    version_file = _version_file(repository)
    original = version_file.read_text(encoding="utf-8")
    _match_version(original, version_file)
    updated = _VERSION_PATTERN.sub(
        lambda match: match.group(0).replace(match.group("version"), version),
        original,
    )
    updates = {version_file: updated, **_studio_versions(repository, version)}
    originals = {path: path.read_text(encoding="utf-8") for path in updates}
    try:
        for path, content in updates.items():
            _write_atomic(path, content)
        check_versions(repository, version)
    except BaseException as error:
        try:
            for path, content in originals.items():
                _write_atomic(path, content)
        except OSError as rollback_error:
            raise RuntimeError(f"Version update failed and rollback was incomplete: {version_file}") from rollback_error
        raise RuntimeError("Version update failed; the previous version was restored.") from error
    return previous_version


def main() -> None:
    """Run the version validation or update command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="New ReMe version, for example 0.4.3")
    parser.add_argument("--check", action="store_true", help="Validate the version without changing files")
    parser.add_argument("--expected-version", help="Also require this release/tag version; a leading v is accepted")
    args = parser.parse_args()
    if args.check:
        if args.version:
            parser.error("version cannot be used with --check; use --expected-version")
        version = check_versions(expected_version=args.expected_version)
        print(f"ReMe release version is valid: {version}")
        return
    if args.expected_version:
        parser.error("--expected-version requires --check")
    if not args.version:
        parser.error("provide a version to update, or use --check")
    previous_version = bump_version(args.version)
    print(f"Updated ReMe from {previous_version} to {args.version}")


if __name__ == "__main__":
    main()
