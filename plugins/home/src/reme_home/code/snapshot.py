"""Bounded navigation of immutable Git objects, independent of the checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys

from ..preparation.code import git, git_bytes, git_tree
from ..preparation.spec import relative_path
from .repository import MAX_OUTPUT, resolve_revision

MAX_FILE_BYTES = 5_000_000
MAX_SEARCH_BYTES = 100_000_000
MAX_READ_LINES = 300


def code_path(path: str) -> str:
    """Accept literal repository-relative paths, never Git pathspec expressions."""
    value = relative_path(path)
    if ".git" in Path(value).parts or "\0" in value:
        raise ValueError("Invalid code path")
    return value


def tree(repository: Path, revision: str, path: str | None = None) -> list[tuple[str, str]]:
    """Read regular-file object IDs with NUL-delimited names and no archive filters."""
    prefix = code_path(path) if path is not None else None

    def selected(name):
        # Browse literal names/prefixes before validating selected paths.
        return prefix is None or name == prefix or name.startswith(prefix + "/")

    return [(name.as_posix(), oid) for name, _mode, oid in git_tree(repository, revision, path_filter=selected)]


def blob(repository: Path, oid: str) -> bytes:
    """Check blob size before loading its bytes; never follow tracked symlinks."""
    if int(git(repository, "cat-file", "-s", oid)) > MAX_FILE_BYTES:
        raise ValueError("Code file exceeds 5 MB; narrow the path")
    return git_bytes(repository, "cat-file", "blob", oid)


def export_blob(repository: Path, oid: str, destination: Path) -> None:
    """Stream a raw Git blob to a new export file, independently of browsing limits."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
    with destination.open("xb") as output:
        subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=" + os.devnull,
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(repository),
                "cat-file",
                "blob",
                oid,
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            check=True,
            env=environment,
            timeout=120,
        )


def list_code_tree(
    repository: Path,
    _record: dict,
    *,
    revision: str = "HEAD",
    path: str | None = None,
    max_results: int = 500,
) -> dict:
    """List tracked paths at a resolved commit."""
    revision = resolve_revision(repository, revision)
    entries = tree(repository, revision, path)
    return {
        "revision": revision,
        "path": path,
        "paths": [name for name, _ in entries[:max_results]],
        "truncated": len(entries) > max_results,
    }


def read_code_file(
    repository: Path,
    _record: dict,
    *,
    path: str,
    revision: str = "HEAD",
    start_line: int = 1,
    end_line: int | None = None,
) -> dict:
    """Return at most 300 lines and 100,000 characters from one committed text file."""
    path = code_path(path)
    revision = resolve_revision(repository, revision)
    entries = dict(tree(repository, revision, path))
    if path not in entries:
        raise ValueError(f"No tracked regular file at {path}")
    data = blob(repository, entries[path])
    if b"\0" in data:
        raise ValueError("Code file is binary")
    lines = data.decode("utf-8", errors="replace").splitlines()
    requested_end = end_line if end_line is not None else start_line + MAX_READ_LINES - 1
    if requested_end < start_line:
        raise ValueError("end_line must be greater than or equal to start_line")
    displayed_end = min(requested_end, start_line + MAX_READ_LINES - 1, len(lines))
    content = "\n".join(lines[start_line - 1 : displayed_end])
    if content:
        content += "\n"
    truncated = len(content) > MAX_OUTPUT or (bool(content) and (start_line > 1 or displayed_end < len(lines)))
    content = content[:MAX_OUTPUT]
    if truncated:
        content += "\n<output truncated; request another line range to continue>"
    return {
        "revision": revision,
        "path": path,
        "start_line": start_line,
        "end_line": displayed_end,
        "total_lines": len(lines),
        "truncated": truncated,
        "content": content,
    }


def search_code(
    repository: Path,
    _record: dict,
    *,
    pattern: str,
    revision: str = "HEAD",
    path: str | None = None,
    max_results: int = 30,
) -> dict:
    """Search with Python multiline regex; isolate regex execution under a hard subprocess timeout."""
    revision = resolve_revision(repository, revision)
    files = {}
    total = 0
    for name, oid in tree(repository, revision, path):
        size = int(git(repository, "cat-file", "-s", oid))
        total += size
        if total > MAX_SEARCH_BYTES:
            raise ValueError("Code search exceeds 100 MB; narrow path")
        data = blob(repository, oid)
        if b"\0" not in data:
            files[name] = data.decode("utf-8", errors="replace")
    # Arbitrary Python regex can backtrack indefinitely. Run only the text matcher
    # in a short-lived stdlib-only process; it never imports or executes candidate code.
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("regex_worker.py"))],
        input=json.dumps({"files": files, "pattern": pattern, "max_results": max_results}).encode(),
        capture_output=True,
        check=True,
        timeout=10,
    )
    payload = json.loads(result.stdout)
    if "error" in payload:
        raise ValueError(payload["error"])
    return {"revision": revision, "pattern": pattern, "path": path, **payload}
