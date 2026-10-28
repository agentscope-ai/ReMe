"""Export trusted source inputs without running their code or installing them."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

from .spec import PreparationSpec, relative_path


def git(repository: Path, *arguments: str) -> str:
    """Run bounded, noninteractive Git commands without a shell or user hooks."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgSign=false", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        env=environment,
        timeout=120,
    )
    return result.stdout.decode().strip()


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
    # Read object contents directly, bypassing archive attributes and checkout filters.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    raw_tree = subprocess.run(
        ["git", "-C", str(source), "ls-tree", "-r", "-z", "--full-tree", revision],
        check=True,
        capture_output=True,
        timeout=120,
        env=environment,
    ).stdout
    roots = [
        Path("reme"),
        Path("pyproject.toml"),
        Path("README.md"),
        Path("LICENSE"),
        *(Path("plugins") / name for name in spec.runtime.plugins),
    ]
    entries = []
    for entry in raw_tree.split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", maxsplit=1)
        name = Path(relative_path(raw_name.decode("utf-8")))
        if not any(name.is_relative_to(root) for root in roots):
            continue
        mode, kind, object_id = header.decode().split()
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError(f"Runtime source must contain regular files only: {name}")
        entries.append((name, mode, object_id))
    objects = subprocess.run(
        ["git", "-C", str(source), "cat-file", "--batch"],
        input="".join(object_id + "\n" for _, _, object_id in entries).encode(),
        check=True,
        capture_output=True,
        timeout=120,
        env=environment,
    ).stdout
    runtime.mkdir()
    candidate.mkdir()
    offset = 0
    for name, mode, object_id in entries:
        boundary = objects.index(b"\n", offset)
        found, kind, size = objects[offset:boundary].decode().split()
        if found != object_id or kind != "blob":
            raise ValueError(f"Invalid Git object: {name}")
        offset = boundary + 1
        # Keep each selected plugin only in repo, including packaging/config files.
        destination = (candidate if name.parts[0] == "plugins" else runtime) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(objects[offset : offset + int(size)])
        destination.chmod(0o755 if mode == "100755" else 0o644)
        offset += int(size) + 1
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
