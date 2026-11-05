"""Local file catalog backend: in-memory dict persisted as compressed JSONL."""

import asyncio
import os
import shutil
import stat
from contextlib import ExitStack
from datetime import date
from pathlib import Path, PureWindowsPath

from .base_file_catalog import BaseFileCatalog
from ..component_registry import R
from ...schema import FileNode
from ...utils.async_utils import complete_in_thread
from ...utils.file_watch import WatchRule, collect_existing, match_file
from ...utils.jsonl_zst import read_jsonl_zst, write_jsonl_zst


@R.register("local")
class LocalFileCatalog(BaseFileCatalog):
    """Dict-backed catalog persisted as JSONL."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self._nodes: dict[str, FileNode] = {}
        self._io_lock = asyncio.Lock()
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        self._catalog_file = self.component_metadata_path / f"{self.name}.jsonl.zst"

    async def load(self) -> None:
        async with self._io_lock:
            if not self._catalog_file.exists():
                return
            await self._read_jsonl()
            self.logger.debug(f"Loaded {len(self._nodes)} nodes from {self._catalog_file}")

    async def dump(self) -> None:
        async with self._io_lock:
            await self._write_jsonl()
            self.logger.info(f"Saved {len(self._nodes)} nodes to {self._catalog_file}")

    async def upsert(self, nodes: list[FileNode]) -> None:
        async with self._io_lock:
            for node in nodes:
                self._nodes[node.path] = node

    async def delete(self, path: str | list[str]) -> None:
        paths = [path] if isinstance(path, str) else path
        async with self._io_lock:
            for p in paths:
                self._nodes.pop(p, None)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        async with self._io_lock:
            if paths is None:
                return list(self._nodes.values())
            return [self._nodes[p] for p in paths if p in self._nodes]

    def _matches(self, path: str, rules: list[WatchRule]) -> bool:
        """Restrict catalog paths to the configured workspace scope."""
        relative = Path(path)
        if not path or relative.anchor or PureWindowsPath(path).anchor or ".." in relative.parts or "\\" in path:
            return False
        if relative.as_posix() != path or path.startswith("~"):
            return False
        absolute = self.workspace_path / relative
        return match_file(str(absolute), rules) and any(
            absolute.is_relative_to(rule.path) and absolute.resolve().is_relative_to(rule.path) for rule in rules
        )

    async def pending(self, limit: int = 10) -> list[str]:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        async with self._io_lock:
            return await complete_in_thread(self._pending, limit)

    def _pending(self, limit: int) -> list[str]:
        rules, today = self.watch_rules, self._today()
        pending = []
        for source, mtime in collect_existing(rules, recursive=True).items():
            path = self.to_workspace_relative(source)
            if not self._matches(path, rules) or self._expired(Path(source), rules, today):
                continue
            node = self._nodes.get(path)
            if node is None or node.st_mtime != mtime:
                pending.append(path)
        return sorted(pending)[:limit]

    async def ack(self, paths: list[str]) -> bool:
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError("paths must be a list of strings")
        async with self._io_lock:
            return await complete_in_thread(self._ack, paths)

    def _ack(self, paths: list[str]) -> bool:
        if not paths:
            return True
        rules, today = self.watch_rules, self._today()
        nodes = dict(self._nodes)
        try:
            for relative in dict.fromkeys(paths):
                path = self.workspace_path / relative
                if not self._matches(relative, rules) or self._expired(path, rules, today):
                    return False
                source_stat = path.stat()
                if not stat.S_ISREG(source_stat.st_mode):
                    return False
                nodes[relative] = FileNode(path=relative, st_mtime=source_stat.st_mtime)
            # Publish in-memory state only after the atomic file replacement succeeds.
            write_jsonl_zst(self._catalog_file, (node.model_dump_json() for node in nodes.values()), self.encoding)
        except OSError as exc:
            self.logger.warning(f"Failed to acknowledge catalog paths: {exc}")
            return False
        self._nodes = nodes
        return True

    async def cleanup_expired(self, directory: str | None = None, ttl_days: int | None = None) -> dict:
        """Remove dated directories at least ``ttl_days`` old, then persist the catalog."""
        ttl_days = self.ttl_days if ttl_days is None else ttl_days
        if not isinstance(ttl_days, int) or isinstance(ttl_days, bool) or ttl_days < 1:
            raise ValueError("ttl_days must be a positive integer")
        directories = (
            [directory]
            if directory is not None
            else [rule.path.relative_to(self.workspace_path).as_posix() for rule in self.watch_rules]
        )
        today = self._today()
        async with self._io_lock:
            # Keep filesystem deletion, catalog mutation and persistence together even on cancellation.
            results = [await complete_in_thread(self._cleanup_expired, root, today, ttl_days) for root in directories]
        return {
            "deleted_dirs": [path for result in results for path in result["deleted_dirs"]],
            "deleted_nodes": sum(result["deleted_nodes"] for result in results),
            "errors": {path: error for result in results for path, error in result["errors"].items()},
        }

    def _cleanup_expired(self, directory: str, today: date, ttl_days: int) -> dict:
        """Inspect source directories, never using cached nodes to authorize deletion."""
        workspace = self.workspace_path.resolve()
        relative_dir = Path(directory)
        root = workspace / relative_dir
        if relative_dir.anchor or str(directory).startswith("~") or ".." in relative_dir.parts or root == workspace:
            raise ValueError("cleanup directory must stay inside the workspace")
        if root.resolve() != root or not root.is_relative_to(workspace):
            raise ValueError("cleanup directory must stay inside the workspace and must not contain symlinks")
        deleted_dirs: list[str] = []
        errors: dict[str, str] = {}
        if root.exists():
            if not shutil.rmtree.avoids_symlink_attacks:
                raise RuntimeError("resource cleanup requires symlink-resistant directory deletion")
            with ExitStack() as stack:
                # Open each path component without following links; keep deletion anchored if the root moves.
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                root_fd = os.open(workspace, flags)
                stack.callback(os.close, root_fd)
                for part in relative_dir.parts:
                    root_fd = os.open(part, flags, dir_fd=root_fd)
                    stack.callback(os.close, root_fd)
                for name in sorted(os.listdir(root_fd)):
                    if not self._expired_date(name, today, ttl_days):
                        continue
                    relative = (relative_dir / name).as_posix()
                    try:
                        if not stat.S_ISDIR(os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_mode):
                            continue
                        shutil.rmtree(name, dir_fd=root_fd)
                    except OSError as exc:
                        errors[relative] = str(exc)
                        self.logger.warning(f"Failed to clean up {relative}: {exc}")
                    else:
                        deleted_dirs.append(relative)

        prefixes = tuple(path + "/" for path in deleted_dirs)
        deleted_nodes: set[str] = set()
        for path in self._nodes:
            absolute = workspace / path
            if absolute.is_relative_to(root):
                parts = absolute.relative_to(root).parts
                missing_expired = (
                    len(parts) > 1 and self._expired_date(parts[0], today, ttl_days) and not (root / parts[0]).exists()
                )
                if path in deleted_dirs or path.startswith(prefixes) or missing_expired:
                    deleted_nodes.add(path)
        nodes = {path: node for path, node in self._nodes.items() if path not in deleted_nodes}
        write_jsonl_zst(self._catalog_file, (n.model_dump_json() for n in nodes.values()), self.encoding)
        self._nodes = nodes
        if deleted_dirs:
            self.logger.info(
                f"Cleaned up {len(deleted_dirs)} dated directories and {len(deleted_nodes)} catalog entries",
            )
        return {"deleted_dirs": deleted_dirs, "deleted_nodes": len(deleted_nodes), "errors": errors}

    async def _read_jsonl(self) -> None:
        for line in read_jsonl_zst(self._catalog_file, self.encoding):
            if stripped := line.strip():
                node = FileNode.model_validate_json(stripped)
                self._nodes[node.path] = node

    async def _write_jsonl(self) -> None:
        write_jsonl_zst(self._catalog_file, (n.model_dump_json() for n in self._nodes.values()), self.encoding)
