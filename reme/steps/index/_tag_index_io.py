"""Safe watch-scope scanning and bounded frontmatter reads for the tag index."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent

from ._watch_rules import WatchRule, build_context_watch_rules

_BOUNDARY = re.compile(rb"^-{3,}\s*$")
_BOM = b"\xef\xbb\xbf"
_CHUNK_BYTES = 4096


@dataclass(frozen=True)
class FrontmatterRead:
    """Structured result of one stable, bounded frontmatter read."""

    status: str
    metadata: dict[str, Any] | None = None
    mtime_ns: int | None = None
    bytes_read: int = 0
    reason: str | None = None


class _NoAliasSafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects aliases before constructing nodes."""

    def compose_node(self, parent, index):
        """Reject alias events, then delegate ordinary node composition."""
        if self.check_event(AliasEvent):
            raise yaml.YAMLError("YAML aliases are disabled")
        return super().compose_node(parent, index)


def validated_watch_rules(app_config, workspace_path: Path, context) -> list[WatchRule]:
    """Build non-empty rules and prove every root stays inside the workspace."""
    if app_config is None:
        raise ValueError("tag index requires application configuration")
    watch_dirs = context.get("watch_dirs", [])
    watch_suffixes = context.get("watch_suffixes", [])
    if not watch_dirs or not watch_suffixes:
        raise ValueError("tag index requires non-empty watch_dirs and watch_suffixes")
    workspace = workspace_path.resolve(strict=True)
    rules = build_context_watch_rules(app_config, workspace, context)
    validated: list[WatchRule] = []
    for rule in rules:
        if "~" in str(rule.path):
            raise ValueError(f"Home-relative watch paths are unsupported: {rule.path}")
        root = rule.path.resolve(strict=False)
        try:
            root.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Watch root escapes workspace: {rule.path}") from exc
        normalized_suffixes = [suffix.strip(".") for suffix in rule.suffixes if suffix.strip(".")]
        if not normalized_suffixes:
            raise ValueError("tag index requires at least one non-empty watch suffix")
        validated.append(WatchRule(path=root, suffixes=normalized_suffixes))
    return validated


def matches_suffix(path: Path, rule: WatchRule) -> bool:
    """Return whether a path has one of a rule's explicit suffixes."""
    return any(path.name.endswith("." + suffix) for suffix in rule.suffixes)


def relative_existing_path(path: str | Path, workspace_path: Path, rules: list[WatchRule]) -> tuple[Path, str]:
    """Resolve an existing regular file and return its safe relative POSIX path."""
    candidate = Path(path)
    if "~" in str(candidate):
        raise ValueError(f"Home-relative paths are unsupported: {path}")
    if not candidate.is_absolute():
        candidate = workspace_path / candidate
    lexical = Path(os.path.abspath(candidate))
    resolved = candidate.resolve(strict=True)
    workspace = workspace_path.resolve(strict=True)
    try:
        relative = lexical.relative_to(workspace).as_posix()
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"File escapes workspace: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"Not a regular file: {path}")
    if not any(
        _within(lexical, rule.path) and _within(resolved, rule.path) and matches_suffix(lexical, rule) for rule in rules
    ):
        raise ValueError(f"File is outside the tag-index watch scope: {path}")
    return resolved, relative


def relative_deleted_path(path: str | Path, workspace_path: Path, rules: list[WatchRule]) -> str:
    """Validate a deleted path lexically, without requiring its target to exist."""
    candidate = Path(path)
    if "~" in str(candidate):
        raise ValueError(f"Home-relative paths are unsupported: {path}")
    if not candidate.is_absolute():
        candidate = workspace_path / candidate
    normalized = Path(os.path.abspath(candidate))
    workspace = workspace_path.resolve(strict=True)
    try:
        relative = normalized.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise ValueError(f"Deleted path escapes workspace: {path}") from exc
    if not any(_within(normalized, rule.path) and matches_suffix(normalized, rule) for rule in rules):
        raise ValueError(f"Deleted path is outside the tag-index watch scope: {path}")
    return relative


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def scan_watch_scope(workspace_path: Path, rules: list[WatchRule], recursive: bool) -> dict[str, tuple[Path, int]]:
    """Fail-closed scan returning relative path -> (resolved path, mtime_ns)."""
    found: dict[str, tuple[Path, int]] = {}

    def onerror(exc: OSError) -> None:
        raise exc

    for rule in rules:
        if not rule.path.is_dir():
            raise RuntimeError(f"Watch root is not a readable directory: {rule.path}")
        if recursive:
            iterator = (
                Path(root) / name
                for root, _dirs, files in os.walk(rule.path, followlinks=False, onerror=onerror)
                for name in files
            )
        else:
            iterator = rule.path.iterdir()
        try:
            for candidate in iterator:
                if not candidate.is_file():
                    continue
                if not matches_suffix(candidate, rule):
                    continue
                resolved, relative = relative_existing_path(candidate, workspace_path, rules)
                found[relative] = (resolved, resolved.stat().st_mtime_ns)
        except OSError as exc:
            raise RuntimeError(f"Could not completely scan watch root {rule.path}: {exc}") from exc
    return found


def read_frontmatter(path: Path, max_bytes: int, retries: int = 2) -> FrontmatterRead:
    """Read a stable frontmatter prefix, retrying twice when the file changes."""
    for attempt in range(retries + 1):
        result, before = _read_frontmatter_once(path, max_bytes)
        if before is None:
            return result
        try:
            after_stat = path.stat()
        except OSError as exc:
            result = FrontmatterRead(status="io_failed", reason=type(exc).__name__, bytes_read=result.bytes_read)
            if attempt < retries:
                continue
            return result
        after = _fingerprint(after_stat)
        if after == before:
            return FrontmatterRead(
                status=result.status,
                metadata=result.metadata,
                mtime_ns=after_stat.st_mtime_ns,
                bytes_read=result.bytes_read,
                reason=result.reason,
            )
        if attempt == retries:
            return FrontmatterRead(status="changed_during_read", reason="fingerprint_changed")
    raise AssertionError("unreachable")


def _fingerprint(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns)


def _read_frontmatter_once(path: Path, max_bytes: int) -> tuple[FrontmatterRead, tuple[int, int, int, int] | None]:
    try:
        before_stat = path.stat()
        before = _fingerprint(before_stat)
        with path.open("rb") as handle:
            buffer = bytearray()
            metadata_bytes = bytearray()
            bytes_read = 0
            opened = False
            while bytes_read < max_bytes:
                chunk = handle.read(min(_CHUNK_BYTES, max_bytes - bytes_read))
                if not chunk:
                    break
                bytes_read += len(chunk)
                buffer.extend(chunk)
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(buffer[:newline]).rstrip(b"\r")
                    del buffer[: newline + 1]
                    if not opened:
                        if line.startswith(_BOM):
                            line = line[len(_BOM) :]
                        if not _BOUNDARY.fullmatch(line):
                            return FrontmatterRead(status="no_frontmatter", bytes_read=bytes_read), before
                        opened = True
                    elif _BOUNDARY.fullmatch(line):
                        return _parse_metadata(bytes(metadata_bytes), bytes_read), before
                    else:
                        metadata_bytes.extend(line)
                        metadata_bytes.extend(b"\n")
            if buffer:
                line = bytes(buffer).rstrip(b"\r")
                if not opened:
                    if line.startswith(_BOM):
                        line = line[len(_BOM) :]
                    if not _BOUNDARY.fullmatch(line):
                        return FrontmatterRead(status="no_frontmatter", bytes_read=bytes_read), before
                    opened = True
                elif _BOUNDARY.fullmatch(line):
                    return _parse_metadata(bytes(metadata_bytes), bytes_read), before
            status = "invalid_frontmatter" if opened else "no_frontmatter"
            reason = "frontmatter_unterminated_or_too_large" if opened else None
            return FrontmatterRead(status=status, bytes_read=bytes_read, reason=reason), before
    except OSError as exc:
        return FrontmatterRead(status="io_failed", reason=type(exc).__name__), None


def _parse_metadata(raw: bytes, bytes_read: int) -> FrontmatterRead:
    try:
        text = raw.decode("utf-8")
        metadata = yaml.load(text, Loader=_NoAliasSafeLoader)  # noqa: S506 - custom SafeLoader subclass
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("frontmatter metadata must be a mapping")
        return FrontmatterRead(status="parsed", metadata=metadata, bytes_read=bytes_read)
    except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        return FrontmatterRead(status="invalid_frontmatter", bytes_read=bytes_read, reason=type(exc).__name__)
