"""Local action-item store: JSONL in the workspace, reloaded into memory."""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .base_action_item_store import BaseActionItemStore
from ..component_registry import R
from ...schema import ActionItem, ActionItemQuery, apply_update


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Serialize everyone touching one store file, across instances/processes.

    Per-instance ``asyncio`` locks coordinate coroutines of one store object
    only. Two stores over the same physical file — another instance, another
    process — each load their own snapshot, so a check-then-replace inside one
    of them can still land on top of a write the other already committed. The
    lock covers the read and the replacement so the pair behaves as one
    operation.

    The lock is a sidecar file opened in place; on POSIX it is an ``flock``,
    on Windows a byte-range lock on the same handle. Both are released when
    the handle closes, so a crashed holder cannot wedge the store.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(handle).st_size == 0:
                os.write(handle, b"\0")
            while True:
                os.lseek(handle, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(handle, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
            try:
                yield
            finally:
                os.lseek(handle, 0, os.SEEK_SET)
                msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def _plain_segment(value: str, label: str) -> str:
    """Reject anything that is not a single directory-entry name.

    Both the store name and the file name are configuration, but a relative
    path with ``..``, an absolute path or a drive prefix silently redirects the
    store outside the workspace: joining an absolute path discards the prefix on
    POSIX and Windows alike.
    """
    if not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty name without surrounding whitespace")
    if value in (".", "..") or any(marker in value for marker in ("/", "\\", ":")):
        raise ValueError(f"{label} must be a plain name, not a path: {value!r}")
    return value


def _plain_file_name(file_name: str) -> str:
    """Reject anything that is not a single file name inside the metadata directory."""
    return _plain_segment(file_name, "file_name")


@R.register("local")
class LocalActionItemStore(BaseActionItemStore):
    """Keep one JSONL file of action items under the workspace metadata directory.

    The file is the durable record: every restart reloads it, and the in-memory
    map is only an index that can always be rebuilt from the file.

    A line this version cannot read is *kept verbatim*. It is skipped in memory
    with a warning and appended unchanged on the next write, so reading data
    written by a newer version - or downgrading afterwards - never destroys it.
    Once a later version understands the line, it is parsed normally and stops
    being preserved by itself.

    If the file itself cannot be read (permissions, invalid bytes), the store
    enters a failed state instead of quietly starting empty: reads raise and
    writes refuse to run, because writing an unread file over itself would
    destroy the only copy of the data.
    """

    def __init__(self, file_name: str = "action_items.jsonl", **kwargs):
        super().__init__(**kwargs)
        self.file_name = _plain_file_name(file_name)
        self._items: dict[str, ActionItem] = {}
        self._alias_index: dict[str, str] = {}
        self._unreadable: list[str] = []
        self._load_failed = False
        # Fingerprint of the file as this instance last saw it. A different value
        # before a write means another instance changed the store underneath us.
        self._file_stamp: str | None = None
        self._io_lock = asyncio.Lock()

    @property
    def _metadata_root(self) -> Path:
        """Resolved per-component metadata directory, proven inside the workspace.

        ``component_metadata_path`` is derived from configuration, so it can be a
        symlink pointing outside the workspace. Anchoring containment to the
        resolved workspace root instead of to that directory keeps every read and
        write inside the workspace even when the metadata tree was redirected.
        """
        workspace = self.workspace_path.resolve()
        root = self.component_metadata_path.resolve()
        if root != workspace and workspace not in root.parents:
            raise ValueError(f"refusing to use a metadata directory outside the workspace: {root}")
        return root

    @property
    def _file(self) -> Path:
        """Location of this store's file: one directory per component name.

        The default file name is shared by every store of this class, so two
        independently named instances used to write the same file and each one
        clobbered the other's snapshot. Scoping the directory by ``name`` gives
        each configured store its own durable record. ``name`` is validated as a
        single path segment because an absolute or relative name would otherwise
        select a directory outside the metadata root.
        """
        return self._metadata_root / _plain_segment(self.name, "store name") / self.file_name

    def _read_target(self) -> Path:
        """The file this store reads, resolved and proven inside the workspace.

        Verifying the metadata root alone is not enough: the named directory
        can itself be a link, and so can the JSONL file inside a regular one.
        Resolving the complete target before opening it means either link is
        caught, not just the root.
        """
        workspace = self.workspace_path.resolve()
        target = self._file.resolve()
        if target != workspace and workspace not in target.parents:
            raise ValueError(
                f"refusing to read a store file outside the workspace: {target}",
            )
        return target

    def _lock_path_for(self, destination: Path) -> Path:
        """Sidecar file that serialises access to *destination*."""
        return destination.with_name(f".{destination.name}.lock")

    @staticmethod
    def _fingerprint(path: Path) -> str | None:
        """Return the file digest, or ``None`` when the file does not exist."""
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except (FileNotFoundError, IsADirectoryError):
            return None

    async def load(self) -> None:
        async with self._io_lock:
            self._load_failed = False
            try:
                target = self._read_target()
            except ValueError as exc:
                self._load_failed = True
                self.logger.error(
                    f"Cannot read {self._file}: {exc}. The store stays unusable until the "
                    "path is repaired, so its contents are never overwritten.",
                )
                raise
            if not target.exists():
                self._items = {}
                self._alias_index = {}
                self._unreadable = []
                self._file_stamp = None
                return

            # One read supplies both the bytes that are parsed and the digest
            # that later guards a write. Re-opening the pathname for the
            # digest could hash a newer version, pairing an old snapshot with
            # a new stamp and letting the next write drop what arrived in
            # between.
            with _exclusive_lock(self._lock_path_for(target)):
                try:
                    raw = target.read_bytes()
                except OSError as exc:
                    self._load_failed = True
                    self.logger.error(
                        f"Cannot read {target}: {exc}. The store stays unusable until the file "
                        "is repaired or removed, so its contents are never overwritten.",
                    )
                    return
            stamp = hashlib.sha256(raw).hexdigest()

            loaded: dict[str, ActionItem] = {}
            unreadable: list[str] = []
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                self._load_failed = True
                self.logger.error(
                    f"Cannot read {target}: {exc}. The store stays unusable until the file "
                    "is repaired or removed, so its contents are never overwritten.",
                )
                return
            for number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = ActionItem.model_validate(json.loads(stripped))
                except (json.JSONDecodeError, ValidationError) as exc:
                    self.logger.warning(
                        f"Kept unreadable action item at {target}:{number} "
                        f"(preserved verbatim, not deleted): {exc}",
                    )
                    if stripped not in unreadable:
                        unreadable.append(stripped)
                    continue
                loaded[item.id] = item

            self._items = loaded
            self._unreadable = unreadable
            self._file_stamp = stamp
            self._reindex()
            if unreadable:
                self.logger.debug(
                    "Loaded %s action items from %s (%s line(s) preserved unread)",
                    len(loaded),
                    self._file,
                    len(unreadable),
                )
            else:
                self.logger.debug("Loaded %s action items from %s", len(loaded), self._file)

    async def dump(self) -> None:
        """Create the file when it is missing; never rewrite an existing one.

        ``upsert`` writes through, so an existing file is already current.
        Rewriting it on close would let a reader instance that only loaded the
        file replace records another instance appended after that load.
        """
        async with self._io_lock:
            if self._file.exists():
                return
            if self._write(self._items, self._unreadable):
                self.logger.info(f"Saved {len(self._items)} action items to {self._file}")
            else:
                self.logger.error(
                    f"Skipped saving action items: {self._file} could not be read, so it is left untouched.",
                )

    def _fail_if_broken(self) -> None:
        if self._load_failed:
            raise RuntimeError(
                f"action-item store is unusable: {self._file} could not be read during load. "
                "Repair or remove the file before using the store.",
            )

    def _build_index(self, items: dict[str, ActionItem]) -> dict[str, str]:
        """Map every historical id to the record currently owning it."""
        index: dict[str, str] = {}
        for item in items.values():
            for alias in item.aliases:
                if alias == item.id or alias in items:
                    continue
                index[alias] = item.id
        return index

    def _reindex(self) -> None:
        self._alias_index = self._build_index(self._items)

    def _related_keys(
        self,
        items: dict[str, ActionItem],
        index: dict[str, str],
        incoming: ActionItem,
    ) -> list[str]:
        """Every stored id this observation describes, in encounter order.

        Both ``incoming.id`` and every alias may already own a record. Ignoring
        the alias path when the id matches leaves two records for one piece of
        work: the alias-linked one keeps its old state forever.
        """
        keys: list[str] = []
        for candidate in [incoming.id, *incoming.aliases]:
            if candidate in items and candidate not in keys:
                keys.append(candidate)
            canonical = index.get(candidate)
            if canonical is not None and canonical in items and canonical not in keys:
                keys.append(canonical)
        return keys

    def _write(self, items: dict[str, ActionItem], unreadable: list[str]) -> bool:
        """Rewrite the file atomically; return ``False`` when the store is unusable.

        Readable records are written in id order, then preserved lines are
        appended exactly as they were read. That keeps the file stable across
        restarts: a line this version cannot read is never dropped, reordered or
        rewritten, and it does not accumulate duplicates because the same single
        line is carried forward each time.

        The caller passes the snapshot to persist rather than the store's live
        map, so a failed write leaves memory untouched. The staging file is
        created exclusively under a random name, so a pre-existing symlink at a
        predictable staging path can never redirect the write outside the
        workspace, and a staging file left by an interrupted write is removed on
        every failure path.
        """
        if self._load_failed:
            return False
        destination = self._file
        self._guard_destination(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._guard_destination(destination)
        hasher = hashlib.sha256()
        descriptor, staged = tempfile.mkstemp(
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(staged)
        try:
            handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
            descriptor = -1
            with handle:
                for item in sorted(items.values(), key=lambda record: record.id):
                    line = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
                    handle.write(line)
                    handle.write("\n")
                    hasher.update(line.encode("utf-8"))
                    hasher.update(b"\n")
                for line in unreadable:
                    handle.write(line)
                    handle.write("\n")
                    hasher.update(line.encode("utf-8"))
                    hasher.update(b"\n")
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        try:
            # Validation and replacement are one operation: a check that ends
            # before another writer commits its own replace cannot see that
            # write, and this one would then drop the records it added.
            with _exclusive_lock(self._lock_path_for(destination)):
                self._assert_unchanged(destination)
                os.replace(temporary, destination)
                self._file_stamp = hasher.hexdigest()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return True

    def _guard_destination(self, destination: Path) -> None:
        """Refuse a destination whose resolved path leaves the allowed tree.

        The parent check alone lets the final component be a link: a regular
        directory holding an ``action_items.jsonl`` symlink still resolves
        outside while its parent looks fine. The parent must stay inside the
        metadata directory, and the fully resolved file inside the workspace.
        """
        base = self._metadata_root
        parent = destination.parent.resolve()
        if parent != base and base not in parent.parents:
            raise ValueError(
                f"refusing to write outside the metadata directory: {destination}",
            )
        workspace = self.workspace_path.resolve()
        resolved = destination.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"refusing to write outside the workspace: {resolved}")

    def _assert_unchanged(self, destination: Path) -> None:
        """Refuse to overwrite a file another instance changed since this load.

        Same-name instances hold independent snapshots, so an unconditional
        replace would drop records the other instance wrote in the meantime.
        Detecting the change and refusing keeps that loss visible instead of
        silent.
        """
        current = self._fingerprint(destination)
        if current != self._file_stamp:
            raise RuntimeError(
                f"refusing to overwrite {destination}: it changed on disk since this instance loaded it. "
                "Reload the store before writing again.",
            )

    def _absorb(
        self,
        items: dict[str, ActionItem],
        index: dict[str, str],
        incoming: ActionItem,
    ) -> ActionItem:
        """Fold one observation into the staged snapshot and return the result.

        When the incoming id and its aliases already own several records, they
        are reconciled into one record: leaving them apart keeps two states for
        the same piece of work, and the alias-linked copy never updates again.
        The oldest ``first_seen`` keeps its id so a stored id never migrates to
        a newer spelling; the others become aliases of it.
        """
        incoming = incoming.model_copy(deep=True)
        keys = self._related_keys(items, index, incoming)
        if not keys:
            items[incoming.id] = incoming
            return incoming

        canonical_key = min(keys, key=lambda key: (items[key].first_seen, key))
        merged = items[canonical_key]
        for key in keys:
            if key == canonical_key:
                continue
            donor = items[key]
            merged = apply_update(merged, donor.model_copy(update={"aliases": [*donor.aliases, donor.id]}))
            del items[key]
        merged = apply_update(merged, incoming)
        items[merged.id] = merged
        return merged

    async def upsert(self, items: list[ActionItem]) -> list[ActionItem]:
        """Merge the observations, persist once, then publish the new snapshot.

        The write is staged: on a failed rename the in-memory map still matches
        the file, so a later query never reports a state the disk rejected.
        """
        async with self._io_lock:
            self._fail_if_broken()
            staged = dict(self._items)
            index = dict(self._alias_index)
            stored: list[ActionItem] = []
            for incoming in items:
                stored.append(self._absorb(staged, index, incoming))
                index = self._build_index(staged)
            if stored:
                self._write(staged, self._unreadable)
                self._items = staged
                self._alias_index = index
            return [item.model_copy(deep=True) for item in stored]

    async def get(self, item_id: str) -> ActionItem | None:
        async with self._io_lock:
            self._fail_if_broken()
            found = self._items.get(item_id)
            if found is None:
                canonical = self._alias_index.get(item_id)
                found = self._items.get(canonical) if canonical is not None else None
            return found.model_copy(deep=True) if found is not None else None

    async def query(self, query: ActionItemQuery | None = None) -> list[ActionItem]:
        filter_ = query or ActionItemQuery()
        async with self._io_lock:
            self._fail_if_broken()
            matched = [item.model_copy(deep=True) for item in self._items.values() if filter_.matches(item)]
        matched.sort(key=lambda item: item.sort_key(), reverse=True)
        if filter_.limit is None:
            return matched
        return matched[: filter_.limit]
