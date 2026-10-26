"""File-backed session store for the Claude Agent SDK."""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk import SessionKey, SessionListSubkeysKey, SessionStoreEntry, SessionStoreListEntry


class CcFileSessionStore:
    """Persist SDK session entries as JSONL under ``project_key/session_id``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _safe_parts(value: str) -> list[str]:
        parts = [part for part in value.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"Invalid session store path component: {value!r}")
        return parts

    def _path(self, *values: str) -> Path:
        path = self.root.joinpath(*(part for value in values for part in self._safe_parts(value)))
        resolved_root = self.root.resolve()
        resolved_path = path.resolve()
        if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
            raise ValueError(f"Session store path escapes root: {resolved_path}")
        return path

    def _project_dir(self, project_key: str) -> Path:
        return self._path(project_key)

    def _path_for_key(self, key: "SessionKey") -> Path:
        values = [key["project_key"], key["session_id"]]
        if subpath := key.get("subpath"):
            values.append(subpath)
        return self._path(*values).with_suffix(".jsonl")

    @staticmethod
    def _read_entries(path: Path) -> list["SessionStoreEntry"]:
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    async def append(self, key: "SessionKey", entries: list["SessionStoreEntry"]) -> None:
        """Append entries while treating their UUIDs as idempotency keys."""
        if not entries:
            return

        path = self._path_for_key(key)
        existing = self._read_entries(path) if path.exists() else []
        seen = {entry.get("uuid") for entry in existing if entry.get("uuid")}
        new_entries = []
        for entry in entries:
            uuid = entry.get("uuid")
            if uuid and uuid in seen:
                continue
            if uuid:
                seen.add(uuid)
            new_entries.append(entry)
        if not new_entries:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            for entry in new_entries:
                file.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    async def load(self, key: "SessionKey") -> list["SessionStoreEntry"] | None:
        """Load all entries for a session or subkey."""
        path = self._path_for_key(key)
        return self._read_entries(path) if path.exists() else None

    async def list_sessions(self, project_key: str) -> list["SessionStoreListEntry"]:
        """List main sessions stored under a project key."""
        project_dir = self._project_dir(project_key)
        if not project_dir.is_dir():
            return []
        return [
            {"session_id": path.stem, "mtime": int(path.stat().st_mtime * 1000)}
            for path in project_dir.glob("*.jsonl")
            if path.is_file()
        ]

    async def delete(self, key: "SessionKey") -> None:
        """Delete one subkey, or a main session and all of its subkeys."""
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()

        if not key.get("subpath"):
            session_dir = self._path(key["project_key"], key["session_id"])
            if session_dir.exists():
                shutil.rmtree(session_dir)

    async def list_subkeys(self, key: "SessionListSubkeysKey") -> list[str]:
        """List subpaths stored below a main session."""
        session_dir = self._path(key["project_key"], key["session_id"])
        if not session_dir.is_dir():
            return []
        return [
            str(path.relative_to(session_dir).with_suffix(""))
            for path in session_dir.rglob("*.jsonl")
            if path.is_file()
        ]
