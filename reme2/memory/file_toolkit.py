"""File category — type-agnostic vault transport + directory operations.

Five tools:

    file_download   copy vault file → session temp dir
    file_upload     copy local file → vault path (any type)
    file_delete     remove vault file (universal entry point)
    file_list       enumerate vault files with optional filters
    file_move       rename / relocate; optional inbound-link rewrite

Session-scoped temp dir is lazy: created on first ``file_download``,
auto-cleaned on process exit. Each download lands in a fresh
sub-directory so concurrent agents can freely modify the temp file
without trampling each other.
"""

from __future__ import annotations

import mimetypes
import shutil
import tempfile
from pathlib import Path

from agentscope.tool import ToolResponse

from . import memory_io
from ..component import R
from ..component.base_step import BaseStep
from .runtime_response import _set_answer, _tool_response


# ===========================================================================
# Section 1 — Helpers (session temp dir + path resolution)
# ===========================================================================


_TEMP_ROOT: Path | None = None


def _get_temp_root() -> Path:
    """Lazy session-scoped temp dir. Auto-cleaned on process exit."""
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp(prefix="reme2-files-"))
    return _TEMP_ROOT


def resolve_vault_path(file_store, vault_path: str) -> Path:
    """Compose the absolute on-disk path for a vault-relative entry.

    Exposed (no underscore) so ``event_toolkit`` can reuse the same
    path-resolution logic when computing event workspace locations.
    """
    working_dir = getattr(file_store, "working_dir", None) or "."
    p = Path(vault_path)
    if p.is_absolute():
        return p.resolve()
    return (Path(working_dir) / p).resolve()


# ===========================================================================
# Section 2 — File tools
# ===========================================================================


@R.register("file_download")
class FileDownload(BaseStep):
    """Copy a vault file to a session temp dir; return the local path."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        vault_path: str = self.context.get("vault_path", "") or ""
        assert vault_path, "vault_path is required"
        payload = self._download(vault_path)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_download(self, vault_path: str) -> ToolResponse:
        """Copy a vault file to session temp dir; return the local path."""
        payload = self._download(vault_path)
        ok = "error" not in payload
        return _tool_response("file_download", ok, payload, audit=self.audit)

    def _download(self, vault_path: str) -> dict:
        src = resolve_vault_path(self.file_store, vault_path)
        if not src.is_file():
            return {"vault_path": vault_path, "error": "not found"}
        dst_dir = Path(tempfile.mkdtemp(prefix="dl-", dir=_get_temp_root()))
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return {
            "vault_path": vault_path,
            "local_path": str(dst),
            "size": dst.stat().st_size,
        }


@R.register("file_upload")
class FileUpload(BaseStep):
    """Copy a local file into the vault. Watcher / parser register the
    FileNode asynchronously."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        local_path: str = self.context.get("local_path", "") or ""
        vault_path: str = self.context.get("vault_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", True))
        assert local_path and vault_path, "local_path and vault_path are required"
        payload = self._upload(local_path, vault_path, overwrite)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_upload(
        self, local_path: str, vault_path: str, overwrite: bool = True,
    ) -> ToolResponse:
        """Copy local_path into the vault at vault_path."""
        payload = self._upload(local_path, vault_path, overwrite)
        ok = "error" not in payload
        return _tool_response("file_upload", ok, payload, audit=self.audit)

    def _upload(self, local_path: str, vault_path: str, overwrite: bool) -> dict:
        src = Path(local_path)
        if not src.is_file():
            return {"local_path": local_path, "error": "source not found"}
        dst = resolve_vault_path(self.file_store, vault_path)
        if dst.exists() and not overwrite:
            return {"vault_path": vault_path, "error": "destination exists; pass overwrite=True"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {
            "vault_path": vault_path,
            "size": dst.stat().st_size,
            "mime": mimetypes.guess_type(dst.name)[0] or "application/octet-stream",
        }


@R.register("file_delete")
class FileDelete(BaseStep):
    """Delete a vault file. Universal entry point for any file type."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        vault_path: str = self.context.get("vault_path", "") or ""
        assert vault_path, "vault_path is required"
        target = resolve_vault_path(self.file_store, vault_path)
        ok, payload = memory_io.delete_file(target)
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def file_delete(self, vault_path: str) -> ToolResponse:
        """Delete a vault file."""
        target = resolve_vault_path(self.file_store, vault_path)
        ok, payload = memory_io.delete_file(target)
        return _tool_response("file_delete", ok, payload, audit=self.audit)


@R.register("file_list")
class FileList(BaseStep):
    """Enumerate vault files with optional frontmatter filters."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        result = memory_io.list_files(
            self.file_store,
            path_prefix=self.context.get("prefix") or self.context.get("path_prefix"),
            tags=self.context.get("tags") or [],
            metadata=self.context.get("metadata") or {},
            limit=int(self.context.get("limit") or 100),
        )
        _set_answer(self.context, result)

    async def file_list(
        self,
        prefix: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        limit: int = 100,
    ) -> ToolResponse:
        """List vault files. Filters: path prefix, frontmatter tags / fields."""
        result = memory_io.list_files(
            self.file_store,
            path_prefix=prefix,
            tags=tags or [],
            metadata=metadata or {},
            limit=limit,
        )
        return _tool_response("file_list", True, result, audit=self.audit)


@R.register("file_move")
class FileMove(BaseStep):
    """Rename / relocate. Default leaves inbound wikilinks untouched
    (maintainer cleans dangling refs); pass ``update_refs=True`` to
    rewrite ``[[old]] → [[new]]`` in every referencing file."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        src: str = self.context.get("src") or self.context.get("old_path") or ""
        dst: str = self.context.get("dst") or self.context.get("new_path") or ""
        update_refs: bool = bool(self.context.get("update_refs", False))
        assert src and dst, "src and dst are required"
        payload = self._move(src, dst, update_refs)
        self.context.response.success = payload.get("ok", False)
        _set_answer(self.context, payload)

    async def file_move(
        self, src: str, dst: str, update_refs: bool = False,
    ) -> ToolResponse:
        """Rename / relocate. update_refs=True rewrites [[old]] → [[new]]."""
        payload = self._move(src, dst, update_refs)
        ok = payload.get("ok", False)
        return _tool_response("file_move", ok, payload, audit=self.audit)

    def _move(self, src: str, dst: str, update_refs: bool) -> dict:
        src_abs = resolve_vault_path(self.file_store, src)
        dst_abs = resolve_vault_path(self.file_store, dst)
        if not src_abs.is_file():
            return {"ok": False, "src": src, "error": "source not found"}
        if update_refs:
            working_dir = Path(getattr(self.file_store, "working_dir", None) or ".").resolve()
            ok, payload = memory_io.rename_file(
                self.file_store, working_dir,
                old_path=src_abs, new_path=dst_abs,
            )
            payload["ok"] = ok
            return payload
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_abs), str(dst_abs))
        return {"ok": True, "src": str(src_abs), "dst": str(dst_abs), "refs_updated": 0}


FILE_TOOL_NAMES: tuple[str, ...] = (
    "file_download",
    "file_upload",
    "file_delete",
    "file_list",
    "file_move",
)
