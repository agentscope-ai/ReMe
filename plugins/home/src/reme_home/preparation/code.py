"""Export trusted source inputs without running their code or installing them."""

import hashlib
from collections.abc import Callable
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

from .spec import PreparationSpec, relative_path


def git_bytes(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    """Run bounded, noninteractive Git commands without a shell or user hooks."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=" + os.devnull,
            "-c",
            "commit.gpgSign=false",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(repository),
            *arguments,
        ],
        check=True,
        capture_output=True,
        input=input_bytes,
        env=environment,
        timeout=120,
    )
    return result.stdout


def git(repository: Path, *arguments: str, strip: bool = True) -> str:
    """Decode Git output, optionally preserving porcelain whitespace and path bytes."""
    output = git_bytes(repository, *arguments).decode("utf-8")
    return output.strip() if strip else output


def runtime_roots(plugins: list[str]) -> list[Path]:
    """Return the core and plugin project roots allowed in prepared sources."""
    return [
        Path("reme"),
        Path("pyproject.toml"),
        Path("README.md"),
        Path("LICENSE"),
        *(Path("plugins") / name for name in plugins),
    ]


def git_tree(repository: Path, revision: str, roots: list[Path] | None = None) -> list[tuple[Path, str, str]]:
    """Select regular Git files using exact NUL-delimited paths, retaining modes and object IDs."""
    entries = []
    raw_tree = git_bytes(repository, "ls-tree", "-r", "-z", "--full-tree", revision)
    for entry in filter(None, raw_tree.split(b"\0")):
        header, raw_name = entry.split(b"\t", 1)
        path = Path(relative_path(raw_name.decode("utf-8")))
        if roots is not None and not any(path.is_relative_to(root) for root in roots):
            continue
        mode, kind, oid = header.decode().split()
        if mode not in {"100644", "100755"} or kind != "blob" or ".git" in path.parts:
            raise ValueError(f"Git source must contain regular files only: {path}")
        entries.append((path, mode, oid))
    return entries


def export_git_tree(
    repository: Path,
    entries: list[tuple[Path, str, str]],
    destination: Callable[[Path], Path],
) -> None:
    """Batch-export raw blobs without checkout/archive filters or browsing size limits."""
    objects = git_bytes(
        repository,
        "cat-file",
        "--batch",
        input_bytes="".join(oid + "\n" for _, _, oid in entries).encode(),
    )
    offset = 0
    for path, mode, oid in entries:
        boundary = objects.index(b"\n", offset)
        found, kind, size = objects[offset:boundary].decode().split()
        if found != oid or kind != "blob":
            raise ValueError(f"Invalid Git object: {path}")
        offset = boundary + 1
        target = destination(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(objects[offset : offset + int(size)])
        target.chmod(0o755 if mode == "100755" else 0o644)
        offset += int(size) + 1


def tree_fingerprint(root: Path) -> str:
    """Hash relative names and file bytes, excluding candidate Git metadata."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic link in prepared source: {relative}")
        if path.is_file():
            digest.update(relative.as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            digest.update(b"\0")
            digest.update(str(path.stat().st_mode & 0o111).encode() + b"\0")
    return digest.hexdigest()


def _check_plugin(root: Path, name: str) -> None:
    project = root / "plugins" / name
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    entries = metadata.get("project", {}).get("entry-points", {}).get("reme.plugins", {})
    if name not in entries:
        raise ValueError(f"Plugin project does not register {name!r}: {project}")
    module = relative_path(entries[name].split(":", maxsplit=1)[0].replace(".", "/"))
    if not any((project / prefix / module / "plugin.yaml").is_file() for prefix in ("src", "")):
        raise ValueError(f"Plugin {name!r} is missing plugin.yaml")


def prepare_code(spec: PreparationSpec, source: Path, runtime: Path, candidate: Path) -> dict:
    """Export fixed ReMe sources and complete, independently versioned plugins.

    Future evaluators must install plugins from the candidate, not the runtime.
    Assemble an isolated evaluation tree from runtime plus candidate (excluding
    .git), then resolve install_projects relative to that tree. candidate_paths
    limits optimization scope; plugins are complete projects even when only src
    is editable. The initial candidate commit supplies the original baseline.
    """
    revision = git(source, "rev-parse", "--verify", "--end-of-options", spec.runtime.source_ref + "^{commit}")
    entries = git_tree(source, revision, runtime_roots(spec.runtime.plugins))
    runtime.mkdir()
    candidate.mkdir()
    # Keep each selected plugin only in repo, including packaging/config files.
    export_git_tree(source, entries, lambda path: (candidate if path.parts[0] == "plugins" else runtime) / path)
    if not (runtime / "reme/__init__.py").is_file() or not (runtime / "pyproject.toml").is_file():
        raise ValueError("Source revision is not a ReMe project")
    if not (runtime / "reme/config" / spec.runtime.config).is_file():
        raise ValueError(f"Runtime configuration does not exist: {spec.runtime.config}")
    for name in spec.runtime.plugins:
        _check_plugin(candidate, name)
    for name in spec.candidate.paths:
        # Plugin candidates already include their full project. Core candidates
        # still copy selected paths from the fixed ReMe runtime for later overlay.
        if (candidate / name).exists():
            continue
        original = runtime / name
        destination = candidate / name
        if not original.exists():
            raise ValueError(f"Candidate path is absent from the selected source: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if original.is_dir():
            shutil.copytree(original, destination)
        else:
            shutil.copy2(original, destination)
    git(candidate, "init", "--initial-branch=init", "--template=")
    git(candidate, "add", "--force", "--all")
    git(candidate, "-c", "user.name=Home", "-c", "user.email=home@localhost", "commit", "-m", "Initial candidate")
    return {
        "source_commit": revision,
        "initial_commit": git(candidate, "rev-parse", "HEAD"),
        "runtime_fingerprint": tree_fingerprint(runtime),
        "candidate_fingerprint": tree_fingerprint(candidate),
        "candidate_paths": spec.candidate.paths,
        "candidate_mount": ".",
        "install_projects": [".", *[f"plugins/{name}" for name in spec.runtime.plugins]],
        "config": spec.runtime.config,
        "plugins": spec.runtime.plugins,
    }
