"""Build deterministic, benchmark-specific ReMe source bundles.

The file allowlists and validation contracts live in ``build_bundle.yaml``.
This module deliberately copies from an allowlist instead of copying the full
repository and pruning it afterwards.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = SCRIPT_DIR / "build_bundle.yaml"
DEFAULT_SOURCE_REPO = SCRIPT_DIR.parent
TARGETS = ("default", "lme", "beam")
_IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


class BundleBuildError(RuntimeError):
    """Raised when a bundle specification or generated bundle is invalid."""


def _load_spec(spec_path: Path) -> Mapping[str, Any]:
    if not spec_path.is_file():
        raise BundleBuildError(f"Bundle specification does not exist: {spec_path}")
    try:
        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BundleBuildError(f"Invalid YAML in {spec_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise BundleBuildError(f"Bundle specification must be a mapping: {spec_path}")
    return loaded


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BundleBuildError(f"'{field}' must be a mapping")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise BundleBuildError(f"'{field}' must be a list of non-empty strings")
    return value


def _target_config(spec: Mapping[str, Any], target: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    targets = _mapping(spec.get("targets"), "targets")
    if target not in targets:
        available = ", ".join(sorted(str(name) for name in targets))
        raise ValueError(f"Unknown bundle target {target!r}; expected one of: {available}")
    common = _mapping(spec.get("common", {}), "common")
    target_spec = _mapping(targets[target], f"targets.{target}")
    return common, target_spec


def _resolve_source_path(source_repo: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BundleBuildError(f"Bundle path must be a safe repository-relative path: {relative!r}")
    resolved = (source_repo / candidate).resolve()
    try:
        resolved.relative_to(source_repo)
    except ValueError as exc:
        raise BundleBuildError(f"Bundle path escapes the source repository: {relative!r}") from exc
    if not resolved.exists():
        raise BundleBuildError(f"Bundle source path does not exist: {relative}")
    return resolved


def _iter_files(source_repo: Path, entries: Iterable[str]) -> list[tuple[Path, Path]]:
    files: dict[Path, Path] = {}
    for entry in entries:
        source = _resolve_source_path(source_repo, entry)
        candidates = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
        for candidate in candidates:
            relative = candidate.relative_to(source_repo)
            if any(part in _IGNORED_PARTS for part in relative.parts) or candidate.suffix in _IGNORED_SUFFIXES:
                continue
            if candidate.is_symlink():
                raise BundleBuildError(f"Symbolic links are not allowed in a bundle: {relative}")
            files[relative] = candidate
    return [(relative, files[relative]) for relative in sorted(files)]


def _write_step_initializers(bundle_root: Path, step_packages: list[str], benchmark_package: str | None) -> None:
    steps_dir = bundle_root / "reme" / "steps"
    imports = list(step_packages)
    if benchmark_package:
        imports.insert(0, "benchmark")
    import_line = f"from . import {', '.join(imports)}\n" if imports else ""
    exported = "\n".join(f'    "{name}",' for name in imports)
    (steps_dir / "__init__.py").write_text(
        f'"""Steps included in this generated bundle."""\n\n{import_line}from .base_step import BaseStep\n\n'
        f'__all__ = [\n    "BaseStep",\n{exported}\n]\n',
        encoding="utf-8",
    )

    if benchmark_package:
        benchmark_dir = steps_dir / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        class_names = {
            "lme": "LmeAgenticAnswerStep, LmeAnswerJudgeStep, LmeAutoMemoryStep",
            "beam": "BeamAgenticAnswerStep, BeamRubricJudgeStep, BeamAutoMemoryStep",
        }
        names = class_names.get(benchmark_package)
        if names is None:
            raise BundleBuildError(f"Unsupported benchmark package: {benchmark_package}")
        exported_names = "\n".join(f'    "{name.strip()}",' for name in names.split(","))
        (benchmark_dir / "__init__.py").write_text(
            '"""Benchmark steps included in this generated bundle."""\n\n'
            f"from . import base, {benchmark_package}\n"
            "from .base import BaseAgenticAnswerStep\n"
            f"from .{benchmark_package} import {names}\n\n"
            f'__all__ = [\n    "BaseAgenticAnswerStep",\n{exported_names}\n    "base",\n'
            f'    "{benchmark_package}",\n]\n',
            encoding="utf-8",
        )


def _merged_validation(common: Mapping[str, Any], target_spec: Mapping[str, Any]) -> dict[str, list[str]]:
    common_validation = _mapping(common.get("validation", {}), "common.validation")
    target_validation = _mapping(target_spec.get("validation", {}), "target.validation")
    result: dict[str, list[str]] = {}
    for field in ("imports", "required_steps", "forbidden_paths"):
        result[field] = _string_list(common_validation.get(field), f"common.validation.{field}") + _string_list(
            target_validation.get(field),
            f"target.validation.{field}",
        )
    return result


def _validate_bundle(bundle_root: Path, validation: Mapping[str, list[str]]) -> None:
    for relative in validation["forbidden_paths"]:
        if (bundle_root / relative).exists():
            raise BundleBuildError(f"Forbidden path was included in bundle: {relative}")

    check = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {str(bundle_root)!r})\n"
        "from reme.components import R\n"
        "from reme.enumeration import ComponentEnum\n"
        f"imports = {validation['imports']!r}\n"
        f"required = {validation['required_steps']!r}\n"
        "for name in imports: importlib.import_module(name)\n"
        "registered = R.get_all(ComponentEnum.STEP)\n"
        "missing = [name for name in required if name not in registered]\n"
        "if missing: raise RuntimeError(f'Missing registered steps: {missing}')\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(bundle_root)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", check],
        cwd=bundle_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise BundleBuildError(f"Generated bundle failed import/registration validation:\n{detail}")


def build_bundle(
    target: str,
    output_dir: Path,
    source_repo: Path = DEFAULT_SOURCE_REPO,
    *,
    spec_path: Path = DEFAULT_SPEC,
    validate: bool = True,
) -> Path:
    """Build one target below ``output_dir/reme`` and return that path."""

    source_repo = Path(source_repo).resolve()
    output_dir = Path(output_dir).resolve()
    spec = _load_spec(Path(spec_path).resolve())
    common, target_spec = _target_config(spec, target)
    entries = _string_list(common.get("include"), "common.include") + _string_list(
        target_spec.get("include"),
        f"targets.{target}.include",
    )
    step_packages = _string_list(common.get("step_packages"), "common.step_packages")
    benchmark_package = target_spec.get("benchmark_package")
    if benchmark_package is not None and not isinstance(benchmark_package, str):
        raise BundleBuildError(f"'targets.{target}.benchmark_package' must be a string")
    files = _iter_files(source_repo, entries)
    validation = _merged_validation(common, target_spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target}-bundle-", dir=output_dir))
    staged_root = temporary / "reme"
    staged_root.mkdir()
    try:
        for relative, source in files:
            destination = staged_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        _write_step_initializers(staged_root, step_packages, benchmark_package)
        if validate:
            _validate_bundle(staged_root, validation)

        destination_root = output_dir / "reme"
        if destination_root.exists():
            if destination_root.is_symlink() or not destination_root.is_dir():
                raise BundleBuildError(f"Refusing to replace non-directory bundle target: {destination_root}")
            shutil.rmtree(destination_root)
        staged_root.replace(destination_root)
        return destination_root
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """Build all bundles from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "bundles")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--no-validate", action="store_true", help="skip import and step registration checks")
    args = parser.parse_args(argv)

    for target in TARGETS:
        bundle = build_bundle(
            target,
            args.output_dir / target,
            args.source_repo,
            spec_path=args.spec,
            validate=not args.no_validate,
        )
        file_count = sum(1 for path in bundle.rglob("*") if path.is_file())
        print(f"{target}: {bundle} ({file_count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
