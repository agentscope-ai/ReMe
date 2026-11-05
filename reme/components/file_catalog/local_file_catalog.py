"""Local file catalog backend: in-memory dict persisted as compressed JSONL."""

import asyncio
import os
import shutil
import stat
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .base_file_catalog import BaseFileCatalog
from ..component_registry import R
from ...schema import FileNode
from ...utils.async_utils import complete_in_thread
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

    async def cleanup_expired(self, directory: str, ttl_days: int) -> dict:
        """Remove dated directories at least ``ttl_days`` old, then persist the catalog."""
        if not isinstance(ttl_days, int) or isinstance(ttl_days, bool) or ttl_days < 1:
            raise ValueError("ttl_days must be a positive integer")
        tz_name = self.app_context.app_config.timezone if self.app_context else None
        today = datetime.now(ZoneInfo(tz_name) if tz_name else None).date()
        async with self._io_lock:
            # Keep filesystem deletion, catalog mutation and persistence together even on cancellation.
            return await complete_in_thread(self._cleanup_expired, directory, today, ttl_days)

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
                    try:
                        day = date.fromisoformat(name)
                    except ValueError:
                        continue
                    if day.isoformat() != name or (today - day).days < ttl_days:
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
        deleted_nodes = [path for path in self._nodes if path in deleted_dirs or path.startswith(prefixes)]
        for path in deleted_nodes:
            del self._nodes[path]
        # Also retry a prior failed write when the source directories have already disappeared.
        write_jsonl_zst(self._catalog_file, (n.model_dump_json() for n in self._nodes.values()), self.encoding)
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
