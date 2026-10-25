"""File-authoritative experience storage with two rebuildable hybrid views."""

import asyncio
import re
from pathlib import Path
from uuid import uuid4

from reme.components.base_component import BaseComponent
from reme.components.file_store import BaseFileStore
from reme.constants import DEFAULT_MAX_FILE_BYTES
from reme.schema import FileChunk, FileNode
from reme.steps.file_io import get_path_lock

from .checkpoint import atomic_json, checked_path, digest, json_text, load_json
from .schema import ProcedureMemory, VIEWS
from .token_budget import TokenBudget
from .trajectory import file_hash, import_file


class ProcedureIndexError(RuntimeError):
    """A durable file exists, but its derived projections need retry or correction."""

    def __init__(self, result: dict):
        self.result = result
        super().__init__(
            f"File committed, indexes pending recovery: {result['path']}. "
            "Retry the same operation, or procedure_read and procedure_update this path "
            "with its current expected_version and a new operation_key. Do not recreate or rename it.",
        )


class ProcedureMemoryStore(BaseComponent):
    """Own single-process locks, import state, journals and immediate index synchronization."""

    component_type = "procedure_memory"

    def __init__(
        self,
        when_to_use_store: str = "procedure_when_to_use",
        when_to_use_content_store: str = "procedure_when_to_use_content",
        procedure_dir: str = "procedures",
        trajectory_subdir: str = "trajectories",
        token_limits: dict | None = None,
        tokenizer_source: str = "Qwen/Qwen3.8-27B",
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.when_to_use_store = self.bind(when_to_use_store, BaseFileStore, optional=False)
        self.when_to_use_content_store = self.bind(when_to_use_content_store, BaseFileStore, optional=False)
        session_dir = self.app_context.app_config.session_dir if self.app_context else "session"
        self.procedure_root = checked_path(self.workspace_path, procedure_dir)
        self.trajectory_root = checked_path(self.workspace_path, f"{session_dir}/{trajectory_subdir}")
        relative_metadata = self.workspace_metadata_path.relative_to(self.workspace_path) / "procedure" / self.name
        self.state_root = checked_path(self.workspace_path, relative_metadata.as_posix())
        roots = [self.procedure_root, self.trajectory_root, self.state_root]
        if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1 :]):
            raise ValueError("Experience, trajectory and metadata directories must not overlap")
        self.budget = TokenBudget(tokenizer_source, **(token_limits or {}))
        if type(max_file_bytes) is not int or max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        self.max_file_bytes = max_file_bytes
        self.processing_lock = asyncio.Lock()
        self.data_lock = asyncio.Lock()
        self.scopes: dict[str, dict] = {}
        self.index_error: str | None = None
        self.rebuild_pending = False
        self.pending_indexes: dict[str, dict] = {}

    @property
    def views(self) -> dict:
        """Resolve the two independent file-store dependencies."""
        return dict(zip(VIEWS, (self.when_to_use_store, self.when_to_use_content_store)))

    async def _start(self) -> None:
        for root in (self.procedure_root, self.trajectory_root, self.state_root):
            root.mkdir(parents=True, exist_ok=True)
        if self.when_to_use_store is self.when_to_use_content_store:
            raise ValueError("The two procedure views must use independent file stores")
        for attribute in ("file_graph", "keyword_index"):
            if getattr(self.when_to_use_store, attribute) is getattr(self.when_to_use_content_store, attribute):
                raise ValueError(f"The two procedure views must use independent {attribute} components")
        for store in self.views.values():
            if not store.embedding_store or not store.keyword_index:
                raise ValueError("Each procedure view requires both embedding_store and keyword_index")
        # A crash can lose cached index dumps, even for an acknowledged write.
        # Rebuild from user files at startup, never from cached chunks alone.
        async with self.data_lock:
            await self._recover_unlocked()
            result = await self._rebuild_unlocked()
            for error in result["errors"]:
                self.logger.warning(f"Procedure rebuild issue: {error}")

    def relative(self, path: Path) -> str:
        """Return a portable path inside this workspace."""
        return path.relative_to(self.workspace_path).as_posix()

    def experience_path(self, relative: str) -> Path:
        """Constrain all experience operations to a direct semantic JSON filename."""
        path = checked_path(self.workspace_path, relative, self.procedure_root)
        if path.parent != self.procedure_root or path.suffix != ".json":
            raise ValueError("Experience must be a direct JSON file in procedure_dir")
        return path

    def source_path(self, relative: str) -> Path:
        """Sources always reference saved trajectory files, never arbitrary host paths."""
        path = checked_path(self.workspace_path, relative, self.trajectory_root)
        if not path.is_file():
            raise ValueError(f"Source trajectory does not exist: {relative}")
        return path

    def _read(self, relative: str) -> dict:
        path = self.experience_path(relative)
        if path.stat().st_size > self.max_file_bytes:
            raise ValueError("Experience exceeds max_file_bytes")
        raw = path.read_bytes()
        memory = ProcedureMemory.model_validate_json(raw)
        for source in memory.sources:
            self.source_path(source)
        return {"path": relative, "version": digest(raw), "experience": memory.document()}

    async def read_experience(self, relative: str) -> dict:
        """Return the full four-field document with an optimistic content version."""
        async with self.data_lock:
            return self._read(relative)

    def import_trajectory(self, file: str, filename: str, request_id: str) -> dict:
        """Called while holding the processing lock; import never overwrites evidence."""
        return import_file(
            self.workspace_path,
            self.trajectory_root,
            self.state_root,
            file,
            filename,
            request_id,
            self.max_file_bytes,
        )

    def open_scope(self, source: str, segment: int) -> str:
        """Opaque runtime capability injected into tools, hidden from the model."""
        self.source_path(source)
        token = uuid4().hex
        self.scopes[token] = {"source": source, "segment": segment}
        return token

    def operation_records(self, source: str, segment: int | None = None) -> list[dict]:
        """Recover acknowledged and in-flight tool context for this segment."""
        records = []
        for path in sorted((self.state_root / "operations").glob("*.json")):
            record = load_json(path)
            if record["source"] == source and (segment is None or record["segment"] == segment):
                records.append(
                    {key: record[key] for key in ("operation_key", "path", "version", "status", "action")},
                )
                for key in ("index_error", "superseded_by"):
                    if key in record:
                        records[-1][key] = record[key]
        return records

    async def apply_mutation(self, action: str, arguments: dict, scope_token: str) -> dict:
        """Journal before write; merge sources and finish both indexes before success."""
        if scope_token not in self.scopes:
            raise PermissionError("Mutation requires an active procedure_auto_memory scope")
        scope = self.scopes[scope_token]
        operation_key = arguments.get("operation_key")
        if not isinstance(operation_key, str) or not operation_key.strip() or len(operation_key) > 200:
            raise ValueError("operation_key must be a nonempty stable key of at most 200 characters")
        signature = digest(json_text({"action": action, "arguments": arguments}))
        operation_path = self.state_root / "operations" / f"{digest(json_text(scope) + operation_key)}.json"
        async with self.data_lock:
            await self._recover_unlocked()
            if operation_path.exists():
                record = load_json(operation_path)
                if record["signature"] != signature:
                    raise ValueError("operation_key already belongs to different arguments; read completed operations")
                return self._operation_result(record)
            record = self._prepare_mutation(action, arguments, scope, signature)
            # Persist the replacement's dependency list before touching either
            # file. Recovery can then skip the old intent even if we crash
            # before marking it superseded or publishing the replacement.
            record["supersedes"] = [
                journal.name
                for journal in (self.state_root / "operations").glob("*.json")
                if (prior := load_json(journal))["path"] == record["path"]
                and prior["status"] not in {"indexed", "superseded"}
            ]
            path = self.experience_path(record["path"])
            async with await get_path_lock(path):
                atomic_json(operation_path, record)
                self._mark_superseded(operation_path, record)
                await self._commit_record(operation_path, record)
            return self._operation_result(record)

    def _prepare_mutation(self, action: str, arguments: dict, scope: dict, signature: str) -> dict:
        if action == "create":
            filename = arguments.get("filename", "")
            if not isinstance(filename, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*\.json", filename):
                raise ValueError("Use a semantic lowercase-hyphenated filename ending in .json")
            relative = self.relative(self.procedure_root / filename)
            path = self.experience_path(relative)
            if path.exists():
                raise ValueError("Experience exists; read it and update, or choose a different meaningful filename")
            previous = None
            document = {key: arguments[key] for key in ("description", "when-to-use", "content")}
            document["sources"] = [scope["source"]]
        elif action == "update":
            relative = arguments["path"]
            existing = self._read(relative)
            previous = arguments["expected_version"]
            if previous != existing["version"]:
                raise ValueError("Version conflict; procedure_read the current experience before retrying")
            changes = arguments["changes"]
            if (
                not isinstance(changes, dict)
                or not changes
                or changes.keys() - {"description", "when-to-use", "content"}
            ):
                raise ValueError("changes may contain only description, when-to-use and content")
            document = {**existing["experience"], **changes}
            document["sources"] = list(dict.fromkeys([*existing["experience"]["sources"], scope["source"]]))
        else:
            raise ValueError("Unknown mutation action")
        memory = ProcedureMemory.model_validate(document)
        for key in ("description", "when-to-use", "content"):
            self.budget.check(key, memory.document()[key])
        for source in memory.sources:
            self.source_path(source)
        payload = memory.document()
        if len(json_text(payload).encode()) > self.max_file_bytes:
            raise ValueError("Experience exceeds max_file_bytes; sources have not been truncated")
        return {
            **scope,
            "operation_key": arguments["operation_key"],
            "signature": signature,
            "action": action,
            "path": relative,
            "previous": previous,
            "document": payload,
            "version": digest(json_text(payload)),
            "status": "prepared",
        }

    async def _commit_record(self, journal: Path, record: dict) -> None:
        path = self.experience_path(record["path"])
        current = digest(path.read_bytes()) if path.exists() else None
        if current != record["version"]:
            if current != record["previous"]:
                raise ValueError(f"Journal conflicts with current experience: {record['path']}")
            atomic_json(path, record["document"])
        record["status"] = "committed"
        atomic_json(journal, record)
        try:
            await self._sync_unlocked([record["path"]])
        except Exception as exc:
            record["index_error"] = str(exc)
            atomic_json(journal, record)
            await self._quarantine_unlocked(record["path"], str(exc))
            raise ProcedureIndexError(
                {**self._operation_result(record), "status": "index_pending", "error": str(exc)},
            ) from exc
        record["status"] = "indexed"
        record.pop("index_error", None)
        atomic_json(journal, record)
        self.pending_indexes.pop(record["path"], None)

    def _mark_superseded(self, replacement: Path, record: dict) -> None:
        """Retire only the older intents named by an already durable replacement."""
        for name in record.get("supersedes", []):
            prior_path = replacement.parent / name
            if prior_path.parent != replacement.parent or prior_path.suffix != ".json":
                raise ValueError("Invalid superseded operation path")
            prior = load_json(prior_path)
            if prior["path"] != record["path"]:
                raise ValueError("A replacement cannot supersede a different experience")
            if prior["status"] != "superseded":
                prior["status"] = "superseded"
                prior["superseded_by"] = record["operation_key"]
                atomic_json(prior_path, prior)

    @staticmethod
    def _operation_result(record: dict) -> dict:
        result = {key: record[key] for key in ("path", "version", "status", "operation_key")}
        if "superseded_by" in record:
            result["superseded_by"] = record["superseded_by"]
        return result

    async def _recover_unlocked(self) -> None:
        journals = sorted((self.state_root / "operations").glob("*.json"))
        # Resolve replacement chains before replay, independent of hashed file
        # ordering and of how far the interrupted replacement had progressed.
        for path in journals:
            self._mark_superseded(path, load_json(path))
        attempted = set()
        for path in journals:
            record = load_json(path)
            if record["status"] not in {"indexed", "superseded"}:
                attempted.add(record["path"])
                try:
                    await self._commit_record(path, record)
                except ProcedureIndexError:
                    pass  # Keep this source file repairable without blocking other files.
        if self.rebuild_pending:
            await self._rebuild_unlocked()
        else:
            for relative in list(self.pending_indexes.keys() - attempted):
                try:
                    await self._sync_unlocked([relative])
                except Exception as exc:
                    await self._quarantine_unlocked(relative, str(exc))
                else:
                    self.pending_indexes.pop(relative, None)

    async def reconcile(self, *, source: str | None = None, segment: int | None = None) -> None:
        """Recover independently; a segment may advance only after its own paths are indexed."""
        async with self.data_lock:
            await self._recover_unlocked()
            if source is not None:
                for record in self.operation_records(source, segment):
                    if record["path"] in self.pending_indexes:
                        raise ProcedureIndexError({**record, **self.pending_indexes[record["path"]]})

    async def _quarantine_unlocked(self, relative: str, error: str, *, persist: bool = True) -> None:
        """Remove only derived records, so neither view serves stale or partial content."""
        self.pending_indexes[relative] = {
            "path": relative,
            "version": self._read(relative)["version"],
            "status": "index_pending",
            "error": error,
        }
        try:
            for store in self.views.values():
                await store.delete(relative)
                if persist:
                    await store.dump()
        except Exception as exc:
            self.index_error = str(exc)
            self.rebuild_pending = True
            raise

    @staticmethod
    def _check_embedding(store) -> None:
        embedding = store.embedding_store
        if not embedding.is_healthy:
            raise RuntimeError("Procedure vector index is unavailable; restore the embedding provider and retry")

    async def _sync_unlocked(self, paths: list[str], *, persist: bool = True) -> None:
        for relative in paths:
            result = self._read(relative)
            memory = ProcedureMemory.model_validate(result["experience"])
            path = self.experience_path(relative)
            for view, store in self.views.items():
                chunk = FileChunk(path=relative, text=memory.projection(view), metadata={"version": result["version"]})
                chunk.set_hash_id()
                node = FileNode(path=relative, st_mtime=path.stat().st_mtime)
                await store.upsert([(node, [chunk])])
                if chunk.embedding is None:
                    self._check_embedding(store)
                    raise RuntimeError(f"Missing vector for {relative} in {view}")
        if persist:
            for store in self.views.values():
                await store.dump()

    async def sync_paths(self, paths: list[str]) -> None:
        """Synchronize complete projections, never chunking an experience."""
        async with self.data_lock:
            await self._recover_unlocked()
            for relative in paths:
                try:
                    await self._sync_unlocked([relative])
                except Exception as exc:
                    await self._quarantine_unlocked(relative, str(exc))
                    raise ProcedureIndexError(self.pending_indexes[relative]) from exc
                self.pending_indexes.pop(relative, None)

    async def _rebuild_unlocked(self) -> dict:
        paths, errors = [], []
        for path in sorted(self.procedure_root.glob("*.json")):
            relative = self.relative(path)
            try:
                self._read(relative)
                paths.append(relative)
            except (OSError, ValueError) as exc:
                errors.append({"path": relative, "error": str(exc)})
        self.rebuild_pending = True
        try:
            for store in self.views.values():
                await store.clear()
            self.pending_indexes.clear()
            for relative in paths:
                try:
                    await self._sync_unlocked([relative], persist=False)
                except Exception as exc:
                    await self._quarantine_unlocked(relative, str(exc), persist=False)
                    errors.append(dict(self.pending_indexes[relative]))
            for store in self.views.values():
                await store.dump()
        except Exception as exc:
            self.index_error = str(exc)
            raise
        self.index_error = None
        self.rebuild_pending = False
        return {"indexed": len(paths) - len(self.pending_indexes), "errors": errors}

    async def rebuild(self) -> dict:
        """Rebuild derived state from experience JSON, leaving invalid user files intact."""
        async with self.data_lock:
            await self._recover_unlocked()
            return await self._rebuild_unlocked()

    async def search(self, query: str, index: str, limit: int = 5, vector_weight: float = 0.7) -> dict:
        """Weighted RRF over one selected hybrid view, returning full current documents."""
        if index not in VIEWS:
            raise ValueError(f"index must be one of {VIEWS}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be an integer between 1 and 200")
        if not 0 <= vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        async with self.data_lock:
            await self._recover_unlocked()
            if self.index_error:
                raise RuntimeError(f"Procedure indexes require recovery or procedure_reindex: {self.index_error}")
            store = self.views[index]
            count = min(200, limit * 5)
            vector, keyword = [], []
            if await store.get_nodes():
                vector = await store.vector_search(query, count, {})
                if not vector:
                    self._check_embedding(store)
                keyword = await store.keyword_search(query, count, {})
            scores = {}
            for hits, weight in ((vector, vector_weight), (keyword, 1 - vector_weight)):
                if weight == 0:
                    continue
                for rank, hit in enumerate(hits, 1):
                    scores[hit.path] = scores.get(hit.path, 0.0) + weight / (60 + rank)
            results = []
            for path in sorted(scores, key=lambda p: (-scores[p], p))[:limit]:
                results.append({**self._read(path), "score": scores[path]})
            return {
                "index": index,
                "results": results,
                "index_complete": not self.pending_indexes,
                "pending_indexes": [self.pending_indexes[path] for path in sorted(self.pending_indexes)],
            }

    def checkpoint_path(self, source: str) -> Path:
        """Checkpoint keyed by the saved source path, without a public trace ID."""
        return self.state_root / "processing" / f"{digest(source)}.json"

    def verify_source(self, source: str, checksum: str) -> None:
        """Reject changed originals rather than resume against different evidence."""
        if file_hash(self.source_path(source), self.max_file_bytes) != checksum:
            raise ValueError("Saved trajectory changed; cannot resume")
