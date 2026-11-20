"""Validated JSON collections with recoverable, version-checked file publication."""

from contextlib import AsyncExitStack
import hashlib
import json
from pathlib import Path

import aiofiles

from ..file_io import check_path_permission, resolve_path, write_file_safe
from ...utils.files.path_lock import get_path_lock
from ...constants import MAX_FILE_READ_BYTES
from ...schema import Idea, Todo

ARCHIVE_LIMIT = 1000


def encode(value):
    """Keep user-owned JSON readable and portable."""
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def fingerprint(content):
    """Distinguish external edits from either side of a pending publication."""
    if content is None:
        return None
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


class RecordStore:
    """One collection; callers hold the workspace notebook lock for each operation."""

    def __init__(self, workspace: Path, notebook_dir: str, kind: str, allowed_paths=None):
        self.workspace = workspace
        self.notebook_dir = notebook_dir
        self.kind = kind
        self.model = Todo if kind == "todos" else Idea
        self.allowed_paths = allowed_paths
        self.files = {}
        self.versions = {}
        self.records = {}
        self.locations = {}

    @property
    def journal(self):
        """A temporary redo intent retained until every affected file is published."""
        return f".{self.kind}.pending.json"

    def path(self, relative):
        """Apply the same containment and request scope as the file jobs."""
        workspace_relative = f"{self.notebook_dir}/{relative}"
        path, error = resolve_path(self.workspace, workspace_relative)
        if error or path is None:
            raise ValueError(error)
        if not check_path_permission(self.workspace, path, self.allowed_paths):
            raise PermissionError(f"no permission to access {workspace_relative}")
        return path

    def valid_file(self, relative):
        """Recovery can publish only this collection's fixed, notebook-relative paths."""
        if relative in {f"{self.kind}.json", f"archive/{self.kind}.json"}:
            return True
        prefix = f"archive/{self.kind}_"
        return relative.startswith(prefix) and relative.endswith(".json") and relative[len(prefix) : -5].isdigit()

    async def read(self, relative):
        """Never treat malformed or truncated source files as empty collections."""
        path = self.path(relative)
        if not path.exists():
            return None
        async with aiofiles.open(path, "rb") as handle:
            data = await handle.read(MAX_FILE_READ_BYTES + 1)
        if len(data) > MAX_FILE_READ_BYTES:
            raise ValueError(f"{relative} exceeds the read limit of {MAX_FILE_READ_BYTES} bytes")
        # json.loads detects Unicode encodings and rejects invalid bytes without replacement.
        return data

    async def recover(self):
        """Finish interrupted writes only while their source versions still match."""
        text = await self.read(self.journal)
        if text is None:
            return
        changes = json.loads(text)
        if not isinstance(changes, dict) or not changes:
            raise ValueError("invalid notebook recovery intent")
        async with AsyncExitStack() as stack:
            for relative, change in sorted(changes.items()):
                if not self.valid_file(relative) or not isinstance(change, dict) or set(change) != {"before", "after"}:
                    raise ValueError("invalid notebook recovery target")
                self.validate(change["after"], relative)
                await stack.enter_async_context(await get_path_lock(self.path(relative)))
            # Preflight every target before publishing any of them.
            for relative, change in changes.items():
                actual = fingerprint(await self.read(relative))
                if actual not in {change["before"], fingerprint(encode(change["after"]))}:
                    raise ValueError(f"{relative} changed outside ReMe; pending operation retained")
            for relative, change in changes.items():
                await write_file_safe(self.path(relative), encode(change["after"]))
            self.path(self.journal).unlink()

    def validate(self, values, relative):
        """Validate every source record, archive placement and local identity."""
        if not isinstance(values, list):
            raise ValueError(f"{relative} must contain a JSON array")
        archived = relative.startswith("archive/")
        if archived and len(values) > ARCHIVE_LIMIT:
            raise ValueError(f"{relative} exceeds {ARCHIVE_LIMIT} archived records")
        records = {}
        for value in values:
            # Defaults are for newly created objects, never for repairing source identities.
            if not isinstance(value, dict) or not value.get("id"):
                raise ValueError(f"{relative} contains a record without an id")
            record = self.model.model_validate(value)
            if isinstance(record, Todo) and any("time" not in note for note in value.get("progress", [])):
                raise ValueError(f"{relative} contains progress without a timestamp")
            closed = (
                record.status in {"done", "cancelled"} if isinstance(record, Todo) else record.closed_at is not None
            )
            if closed != archived or (record.closed_at is not None) != archived:
                raise ValueError(f"{relative} contains a record with inconsistent archive state")
            if record.id in records:
                raise ValueError(f"{relative} contains duplicate id {record.id}")
            records[record.id] = record
        return records

    async def load(self, *, archives=True):
        """Read authoritative arrays rather than relying on an asynchronous search index."""
        await self.recover()
        paths = [f"{self.kind}.json"]
        if archives:
            directory = self.path("archive")
            paths.append(f"archive/{self.kind}.json")
            if directory.exists():
                shards = [f"archive/{path.name}" for path in directory.glob(f"{self.kind}_*.json")]
                paths.extend(
                    sorted(
                        (path for path in shards if self.valid_file(path)),
                        key=lambda p: int(p.rsplit("_", 1)[1][:-5]),
                    ),
                )
        for relative in paths:
            text = await self.read(relative)
            records = self.validate(json.loads(text) if text is not None else [], relative)
            self.files[relative] = records
            self.versions[relative] = fingerprint(text)
            for identifier, record in records.items():
                if identifier in self.records:
                    raise ValueError(f"duplicate {self.kind} id {identifier} across files")
                self.records[identifier] = record
                self.locations[identifier] = relative
        return self

    def get(self, identifier):
        """Resolve stable IDs across active and archived arrays."""
        if identifier not in self.records:
            raise ValueError(f"{self.kind} record not found: {identifier}")
        return self.records[identifier]

    async def put(self, record):
        """Publish one record and any required archive migration as a redoable unit."""
        active = f"{self.kind}.json"
        previous = self.locations.get(record.id)
        archived = record.closed_at is not None
        if not archived:
            target = active
        elif previous and previous != active:
            target = previous
        else:
            archives = [path for path in self.files if path.startswith("archive/")]
            target = archives[-1] if archives else f"archive/{self.kind}.json"
            if len(self.files.get(target, {})) >= ARCHIVE_LIMIT:
                number = int(target.rsplit("_", 1)[1][:-5]) + 1 if "_" in Path(target).stem else 1
                target = f"archive/{self.kind}_{number:04d}.json"
        changes = {}
        for relative in {previous, target} - {None}:
            records = dict(self.files.get(relative, {}))
            records.pop(record.id, None)
            if relative == target:
                records[record.id] = record
            values = [item.model_dump(mode="json", exclude_none=True) for item in records.values()]
            self.validate(values, relative)
            changes[relative] = {"before": self.versions.get(relative), "after": values}
        intent = encode(changes)
        if len(intent.encode("utf-8")) > MAX_FILE_READ_BYTES:
            raise ValueError("notebook update exceeds the recoverable read limit")
        self.path(self.journal)
        async with AsyncExitStack() as stack:
            for relative in sorted(changes):
                await stack.enter_async_context(await get_path_lock(self.path(relative)))
            for relative, change in changes.items():
                if fingerprint(await self.read(relative)) != change["before"]:
                    raise ValueError(f"{relative} changed outside ReMe; reload before updating")
            await write_file_safe(self.path(self.journal), intent)
        await self.recover()
