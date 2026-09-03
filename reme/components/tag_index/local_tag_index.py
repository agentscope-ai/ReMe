"""Local dict/set-backed tag index with an atomic JSONL/zstd snapshot."""

import asyncio
from pathlib import PurePosixPath

from .base_tag_index import BaseTagIndex
from ..component_registry import R
from ...schema import TagFileRecord, TagIndexSnapshot, TagRecord, TagSourceRecord
from ...utils.async_utils import complete_in_thread
from ...utils.jsonl_zst import read_jsonl_zst, write_jsonl_zst


@R.register("local")
class LocalTagIndex(BaseTagIndex):
    """Maintain bidirectional file/tag relationships using positive integer IDs."""

    def __init__(
        self,
        max_tags_per_file: int = 8,
        max_tag_length: int = 64,
        max_frontmatter_bytes: int = 65_536,
        index_version: str = "v1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_tags_per_file = self._positive_int("max_tags_per_file", max_tags_per_file)
        self.max_tag_length = self._positive_int("max_tag_length", max_tag_length)
        self.max_frontmatter_bytes = self._positive_int("max_frontmatter_bytes", max_frontmatter_bytes)
        self.index_version = str(index_version)
        if not self.index_version:
            raise ValueError("index_version must not be empty")
        self.index_file = self.component_metadata_path / f"tag_index_{self.name}_{self.index_version}.jsonl.zst"
        self._maintenance_lock = asyncio.Lock()
        self.loaded = False
        self._state_persistable = True
        self.next_file_id: int
        self.next_tag_id: int
        self.files: dict[int, TagFileRecord]
        self.tags: dict[int, tuple[str, set[int]]]
        self.path_to_file_id: dict[str, int]
        self.tag_to_tag_id: dict[str, int]
        self._reset_state()

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _reset_state(self) -> None:
        self.next_file_id = 1
        self.next_tag_id = 1
        self.files: dict[int, TagFileRecord] = {}
        self.tags: dict[int, tuple[str, set[int]]] = {}
        self.path_to_file_id: dict[str, int] = {}
        self.tag_to_tag_id: dict[str, int] = {}

    async def _start(self) -> None:
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        await self.load()

    async def _close(self) -> None:
        if self.is_started and self._state_persistable:
            await self.dump()

    def normalize_tags(self, value: object) -> list[str]:
        """Normalize a strict YAML-list tags value into canonical strings."""
        if not isinstance(value, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                continue
            raw = str(item).strip()
            if not raw or len(raw) > self.max_tag_length or any(char.isspace() for char in raw):
                continue
            if not any(char.isalnum() for char in raw):
                continue
            canonical = raw.casefold()
            if canonical in seen:
                continue
            seen.add(canonical)
            result.append(canonical)
            if len(result) >= self.max_tags_per_file:
                break
        return result

    @staticmethod
    def _validate_path(path: str) -> str:
        if not isinstance(path, str) or not path or "\\" in path:
            raise ValueError(f"Invalid workspace-relative tag-index path: {path!r}")
        pure = PurePosixPath(path)
        if pure.is_absolute() or path != pure.as_posix() or any(part in ("", ".", "..") for part in pure.parts):
            raise ValueError(f"Invalid workspace-relative tag-index path: {path!r}")
        return path

    async def get_records(self) -> list[TagSourceRecord]:
        async with self._maintenance_lock:
            return [
                TagSourceRecord(
                    path=record.path,
                    mtime_ns=record.mtime_ns,
                    tags=[self.tags[tag_id][0] for tag_id in record.tag_ids],
                )
                for _, record in sorted(self.files.items())
            ]

    async def upsert(self, records: list[TagSourceRecord]) -> None:
        async with self._maintenance_lock:
            for record in records:
                self._upsert(record)
            self._validate_live()

    async def delete(self, paths: list[str]) -> None:
        async with self._maintenance_lock:
            for path in paths:
                self._delete(self._validate_path(path))
            self._validate_live()

    async def reconcile(
        self,
        records: list[TagSourceRecord],
        deleted_paths: list[str],
        *,
        rebuild: bool = False,
        persist: bool = True,
    ) -> None:
        """Atomically apply one prepared audit/reconcile commit."""
        async with self._maintenance_lock:
            previous = self._snapshot()
            previous_persistable = self._state_persistable
            self._state_persistable = False
            try:
                if rebuild:
                    self._reset_state()
                for path in deleted_paths:
                    self._delete(self._validate_path(path))
                for record in records:
                    self._upsert(record)
                self._validate_live()
                self._state_persistable = True
                if persist:
                    await self._dump_locked()
            except Exception:
                self._install_candidate(previous)
                self._state_persistable = previous_persistable
                raise

    def _upsert(self, source: TagSourceRecord) -> None:
        path = self._validate_path(source.path)
        if isinstance(source.mtime_ns, bool) or source.mtime_ns < 0:
            raise ValueError("mtime_ns must be a non-negative integer")
        names = self.normalize_tags(source.tags)
        file_id = self.path_to_file_id.get(path)
        if file_id is None:
            file_id = self.next_file_id
            self.next_file_id += 1
            old_tag_ids: set[int] = set()
        else:
            old_tag_ids = set(self.files[file_id].tag_ids)

        new_tag_ids: list[int] = []
        for name in names:
            tag_id = self.tag_to_tag_id.get(name)
            if tag_id is None:
                tag_id = self.next_tag_id
                self.next_tag_id += 1
                self.tags[tag_id] = (name, set())
                self.tag_to_tag_id[name] = tag_id
            new_tag_ids.append(tag_id)

        new_tag_set = set(new_tag_ids)
        for tag_id in old_tag_ids - new_tag_set:
            name, file_ids = self.tags[tag_id]
            file_ids.discard(file_id)
            if not file_ids:
                del self.tags[tag_id]
                del self.tag_to_tag_id[name]
        for tag_id in new_tag_set - old_tag_ids:
            self.tags[tag_id][1].add(file_id)

        self.files[file_id] = TagFileRecord(id=file_id, path=path, mtime_ns=source.mtime_ns, tag_ids=new_tag_ids)
        self.path_to_file_id[path] = file_id

    def _delete(self, path: str) -> None:
        file_id = self.path_to_file_id.pop(path, None)
        if file_id is None:
            return
        record = self.files.pop(file_id)
        for tag_id in record.tag_ids:
            name, file_ids = self.tags[tag_id]
            file_ids.discard(file_id)
            if not file_ids:
                del self.tags[tag_id]
                del self.tag_to_tag_id[name]

    def _snapshot(self) -> TagIndexSnapshot:
        return TagIndexSnapshot(
            next_file_id=self.next_file_id,
            next_tag_id=self.next_tag_id,
            files=[self.files[file_id].model_copy(deep=True) for file_id in sorted(self.files)],
            tags=[
                TagRecord(id=tag_id, name=self.tags[tag_id][0], file_ids=sorted(self.tags[tag_id][1]))
                for tag_id in sorted(self.tags)
            ],
            max_tag_length=self.max_tag_length,
            max_tags_per_file=self.max_tags_per_file,
            max_frontmatter_bytes=self.max_frontmatter_bytes,
        )

    async def dump(self) -> None:
        async with self._maintenance_lock:
            if not self._state_persistable:
                raise RuntimeError("Refusing to persist an incomplete tag-index state")
            await self._dump_locked()

    async def _dump_locked(self) -> None:
        if not self.files:
            self.index_file.unlink(missing_ok=True)
            return
        snapshot = self._snapshot()
        await complete_in_thread(write_jsonl_zst, self.index_file, [snapshot.model_dump_json()])
        self.logger.info(f"Saved {len(self.files)} files and {len(self.tags)} tags to {self.index_file}")

    async def load(self) -> bool:
        async with self._maintenance_lock:
            self.loaded = False
            self._reset_state()
            if not self.index_file.exists():
                return False
            try:
                snapshot = await asyncio.to_thread(self._read_snapshot)
                candidate = self._validate_snapshot(snapshot)
                self._install_candidate(candidate)
                self.loaded = True
                self._state_persistable = True
                self.logger.info(f"Loaded {len(self.files)} files and {len(self.tags)} tags from {self.index_file}")
                return True
            except Exception as exc:
                self.logger.exception(f"Failed to load tag index; rebuilding on next sync: {exc}")
                self.index_file.unlink(missing_ok=True)
                self._reset_state()
                return False

    def _read_snapshot(self) -> TagIndexSnapshot:
        lines = [line.strip() for line in read_jsonl_zst(self.index_file) if line.strip()]
        if len(lines) != 1:
            raise ValueError("tag-index snapshot must contain exactly one JSONL record")
        return TagIndexSnapshot.model_validate_json(lines[0])

    def _validate_snapshot(self, snapshot: TagIndexSnapshot) -> TagIndexSnapshot:
        for name in ("next_file_id", "next_tag_id"):
            self._positive_int(name, getattr(snapshot, name))
        expected = (self.max_tag_length, self.max_tags_per_file, self.max_frontmatter_bytes)
        actual = (snapshot.max_tag_length, snapshot.max_tags_per_file, snapshot.max_frontmatter_bytes)
        if actual != expected:
            raise ValueError(f"tag-index parameters changed: expected {expected}, got {actual}")

        file_ids: set[int] = set()
        paths: set[str] = set()
        file_map: dict[int, TagFileRecord] = {}
        for record in snapshot.files:
            self._positive_int("file id", record.id)
            if record.id in file_ids or record.path in paths:
                raise ValueError("duplicate file id or path")
            self._validate_path(record.path)
            if isinstance(record.mtime_ns, bool) or not isinstance(record.mtime_ns, int) or record.mtime_ns < 0:
                raise ValueError("invalid mtime_ns")
            if len(record.tag_ids) > self.max_tags_per_file or len(record.tag_ids) != len(set(record.tag_ids)):
                raise ValueError("invalid file tag_ids")
            file_ids.add(record.id)
            paths.add(record.path)
            file_map[record.id] = record

        tag_ids: set[int] = set()
        names: set[str] = set()
        tag_map: dict[int, TagRecord] = {}
        for record in snapshot.tags:
            self._positive_int("tag id", record.id)
            if record.id in tag_ids or record.name in names:
                raise ValueError("duplicate tag id or name")
            if self.normalize_tags([record.name]) != [record.name]:
                raise ValueError(f"non-canonical tag name: {record.name!r}")
            if not record.file_ids or len(record.file_ids) != len(set(record.file_ids)):
                raise ValueError("tag posting must be non-empty and unique")
            tag_ids.add(record.id)
            names.add(record.name)
            tag_map[record.id] = record

        if snapshot.next_file_id <= max(file_ids, default=0) or snapshot.next_tag_id <= max(tag_ids, default=0):
            raise ValueError("next id must exceed every live id")
        for file_id, file_record in file_map.items():
            for tag_id in file_record.tag_ids:
                if tag_id not in tag_map or file_id not in tag_map[tag_id].file_ids:
                    raise ValueError("broken file-to-tag relationship")
        for tag_id, tag_record in tag_map.items():
            for file_id in tag_record.file_ids:
                if file_id not in file_map or tag_id not in file_map[file_id].tag_ids:
                    raise ValueError("broken tag-to-file relationship")
        return snapshot

    def _install_candidate(self, snapshot: TagIndexSnapshot) -> None:
        self._reset_state()
        self.next_file_id = snapshot.next_file_id
        self.next_tag_id = snapshot.next_tag_id
        self.files = {record.id: record.model_copy(deep=True) for record in snapshot.files}
        self.tags = {record.id: (record.name, set(record.file_ids)) for record in snapshot.tags}
        self.path_to_file_id = {record.path: record.id for record in snapshot.files}
        self.tag_to_tag_id = {record.name: record.id for record in snapshot.tags}
        self._validate_live()

    def _validate_live(self) -> None:
        self._validate_snapshot(self._snapshot())

    async def clear(self) -> None:
        async with self._maintenance_lock:
            self._reset_state()
            self.loaded = False
            self._state_persistable = True
            self.index_file.unlink(missing_ok=True)
