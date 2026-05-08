"""Abstract base class for file stores.

A `FileStore` manages **everything** about a file in the vault:
  - file metadata (frontmatter + mtime)
  - the wikilink graph: edges per file (links / backlinks / resolution)
  - graph-walk retrieval primitives (BFS expansion, decay scoring)
  - chunks (text + embeddings + position) with vector / keyword / hybrid search

Edges are stored independently of metadata in `_edges: dict[path,
list[FileEdge]]`. The watcher submits a `ParsedFile` via
`upsert_parsed(parsed)`; the store fans this out to meta + edge + chunk
persistence in a single atomic-from-the-caller's-POV operation.

Reads (`get_file_meta`, `get_links`, `resolve_wikilink`, `subgraph_score`,
`filter`, ...) are **synchronous** — they hit an in-memory index that is
rebuilt from the persistent backend on `_start`. Writes are **async** —
they touch persistence first, then update the in-memory index, so on
crash the on-disk view is the source of truth.

Backends supply persistence for chunks, file metadata, AND edges via
abstract methods. The default meta + edge persistence is sidecar JSONL;
SQLite overrides with relational tables.
"""

import re
from abc import abstractmethod
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ...enumeration import ComponentEnum
from ...schema import ChunkFilter, FileChunk, FileEdge, FileMetadata, ParsedFile
from ...utils.wikilink import extract_wikilinks


def _target_stem(raw: str) -> str:
    """Stem of a raw wikilink target — `"topics/X/X"` → `"X"`."""
    target = raw.strip()
    if target.endswith(".md"):
        target = target[:-3]
    return target.rsplit("/", 1)[-1]


class BaseFileStore(BaseComponent):
    """Unified file store: metadata + edges + chunks + graph + search."""

    component_type = ComponentEnum.FILE_STORE

    def __init__(
        self,
        store_name: str,
        db_path: str | Path,
        embedding_model: str = "default",
        fts_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._embedding_model_name: str = embedding_model
        self.embedding_model: BaseEmbeddingModel | None = None
        self.store_name: str = store_name
        self.db_path: Path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vector_enabled: bool = bool(embedding_model)
        self.fts_enabled: bool = fts_enabled

        if not re.match(r"^[a-zA-Z0-9_]+$", store_name):
            raise ValueError(
                f"Invalid store name '{store_name}'. Only alphanumeric characters and underscores are allowed.",
            )
        if not self.vector_enabled and not self.fts_enabled:
            raise ValueError("At least one of embedding_model or fts_enabled must be set.")

        # In-memory indices (rebuilt on _start from persisted state).
        self._nodes: dict[str, FileMetadata] = {}
        self._edges: dict[str, list[FileEdge]] = {}
        self._stems: dict[str, set[str]] = defaultdict(set)
        self._backlinks: dict[str, set[str]] = defaultdict(set)
        self.vault_root: Path | None = None

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self):
        if self._embedding_model_name:
            assert self.app_context is not None, "app_context must be provided"
            models = self.app_context.components.get(ComponentEnum.EMBEDDING_MODEL, {})
            if self._embedding_model_name not in models:
                raise ValueError(f"Embedding model '{self._embedding_model_name}' not found.")
            model = models[self._embedding_model_name]
            if not isinstance(model, BaseEmbeddingModel):
                raise TypeError(f"Expected BaseEmbeddingModel, got {type(model).__name__}")
            self.embedding_model = model
        # Concrete backend's _start runs first (opens conn, creates tables);
        # this base _start runs LAST so persistence is ready by the time we
        # iterate it. Subclasses should `await super()._start()` at the END
        # of their own _start.
        await self._reload_file_metas()

    async def _close(self):
        self.embedding_model = None
        self._nodes.clear()
        self._edges.clear()
        self._stems.clear()
        self._backlinks.clear()

    async def _reload_file_metas(self) -> None:
        """Rebuild in-memory indices from persisted meta + edges. Called on _start."""
        self._nodes.clear()
        self._edges.clear()
        self._stems.clear()
        self._backlinks.clear()
        count = 0
        for meta in self._iter_persisted_metas():
            self._nodes[meta.path] = meta
            self._stems[Path(meta.path).stem].add(meta.path)
            count += 1
        edge_count = 0
        for path, edges in self._iter_persisted_edges():
            if path not in self._nodes:
                continue  # orphan — meta was already pruned
            self._edges[path] = edges
            for edge in edges:
                self._backlinks[_target_stem(edge.target)].add(path)
                edge_count += 1
        self.logger.info(
            f"[{self.store_name}] Loaded {count} file metas, {edge_count} edges into in-memory index",
        )

    # -- Embedding helpers --------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self.embedding_model.dimensions if self.embedding_model else 1024

    def _disable_vector_search(self, reason: str = "embedding API error") -> None:
        if self.vector_enabled:
            self.logger.warning(f"[{self.store_name}] Disabling vector search: {reason}")
            self.vector_enabled = False

    async def _get_embeddings_safe(self, texts: list[str], **kwargs) -> list[list[float]] | None:
        if not self.vector_enabled:
            return None
        try:
            assert self.embedding_model is not None
            return await self.embedding_model.get_embeddings(texts, **kwargs)
        except Exception as e:
            self._disable_vector_search(str(e))
            return None

    async def get_embedding(self, query: str, **kwargs) -> list[float] | None:
        result = await self._get_embeddings_safe([query], **kwargs)
        return result[0] if result else None

    async def get_embeddings(self, queries: list[str], **kwargs) -> list[list[float]] | None:
        return await self._get_embeddings_safe(queries, **kwargs)

    # -- Vault root ---------------------------------------------------------

    def set_vault_root(self, vault_root: str | Path | None) -> None:
        """Bind the runtime vault root for explicit-path wikilink resolution."""
        self.vault_root = Path(vault_root).resolve() if vault_root is not None else None

    # -- File-meta CRUD (async writes; sync reads on in-memory index) -------

    async def upsert_file_meta(self, meta: FileMetadata) -> None:
        """Persist + index a file meta. Replaces any prior entry for the path.

        Edges are NOT touched here — call `upsert_edges` separately, or
        use `upsert_parsed` for the combined fan-out from a `ParsedFile`.
        """
        await self._persist_upsert_meta(meta)
        self._nodes[meta.path] = meta
        self._stems[Path(meta.path).stem].add(meta.path)

    async def update_file_meta(self, path: str, **fields) -> FileMetadata | None:
        """Patch an existing meta (model_copy + upsert). No-op if path unknown."""
        existing = self._nodes.get(path)
        if existing is None:
            return None
        updated = existing.model_copy(update=fields)
        await self.upsert_file_meta(updated)
        return updated

    async def delete_file_meta(self, path: str) -> FileMetadata | None:
        """Persist deletion + drop from index. Also drops the file's edges.

        Returns prior meta if any.
        """
        await self._persist_delete_meta(path)
        await self._persist_delete_edges(path)
        existing = self._nodes.pop(path, None)
        if existing is not None:
            stem = Path(existing.path).stem
            self._stems.get(stem, set()).discard(existing.path)
            if not self._stems.get(stem):
                self._stems.pop(stem, None)
        for edge in self._edges.pop(path, []):
            tgt = _target_stem(edge.target)
            self._backlinks.get(tgt, set()).discard(path)
            if not self._backlinks.get(tgt):
                self._backlinks.pop(tgt, None)
        return existing

    def get_file_meta(self, path: str) -> FileMetadata | None:
        return self._nodes.get(path)

    @property
    def nodes(self) -> dict[str, FileMetadata]:
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, path: str) -> bool:
        return path in self._nodes

    # -- Edge CRUD ----------------------------------------------------------

    async def upsert_edges(self, path: str, edges: list[FileEdge]) -> None:
        """Persist + reindex the edge list for `path`. Replaces any prior set."""
        await self._persist_upsert_edges(path, edges)
        # Drop old backlink contributions from this source.
        for edge in self._edges.get(path, []):
            tgt = _target_stem(edge.target)
            self._backlinks.get(tgt, set()).discard(path)
            if not self._backlinks.get(tgt):
                self._backlinks.pop(tgt, None)
        if edges:
            self._edges[path] = list(edges)
            for edge in edges:
                self._backlinks[_target_stem(edge.target)].add(path)
        else:
            self._edges.pop(path, None)

    def get_edges(self, path: str) -> list[FileEdge]:
        """All edges originating from `path` (raw — not resolved)."""
        return list(self._edges.get(path, ()))

    async def upsert_parsed(self, parsed: ParsedFile) -> None:
        """Single fan-out entry point for the watcher's parse pass.

        Persists meta, edges, then chunks — the watcher gets one call and
        the store handles the three sub-payloads in order.
        """
        await self.upsert_file_meta(FileMetadata(
            file=parsed.file,
            path=parsed.path,
            st_mtime=parsed.st_mtime,
            metadata=parsed.metadata,
        ))
        await self.upsert_edges(parsed.path, parsed.edges)
        await self.upsert_chunks(parsed.path, parsed.chunks)

    # -- Link queries (sync, in-memory) -------------------------------------

    def get_links(self, path: str) -> list[tuple[FileMetadata, FileEdge]]:
        """Files this `path` links TO (resolved only).

        Returns a list of `(target_meta, edge)` pairs — predicate / anchor /
        source / confidence are preserved on the edge so callers can reason
        about typed graph structure. The same target file can appear multiple
        times if linked via several edges (e.g. plain `[[X]]` plus
        `[author:: [[X]]]`); caller dedupes by `target_meta.path` if needed.
        """
        if path not in self._nodes:
            return []
        out: list[tuple[FileMetadata, FileEdge]] = []
        for edge in self._edges.get(path, ()):
            hit = self.resolve_wikilink(edge.target)
            if hit is not None and hit in self._nodes:
                out.append((self._nodes[hit], edge))
        return out

    def get_backlinks(self, path: str) -> list[tuple[FileMetadata, FileEdge]]:
        """Files that link TO `path` (resolved only).

        Returns a list of `(source_meta, edge)` pairs — `edge` is the
        specific FileEdge on the source file that resolves to `path`. A
        single source can appear multiple times if it points at `path` via
        multiple edges.
        """
        if path not in self._nodes:
            return []
        stem = Path(path).stem
        candidates = self._backlinks.get(stem, set())
        out: list[tuple[FileMetadata, FileEdge]] = []
        for src in candidates:
            src_meta = self._nodes.get(src)
            if src_meta is None:
                continue
            for edge in self._edges.get(src, ()):
                if self.resolve_wikilink(edge.target) == path:
                    out.append((src_meta, edge))
        return out

    def get_paths_by_stem(self, stem: str) -> list[str]:
        return sorted(self._stems.get(stem, set()))

    # -- Wikilink resolution ------------------------------------------------

    def resolve_wikilink(self, wikilink: str) -> str | None:
        target = wikilink.strip()
        if not target:
            return None

        if "/" in target or target.endswith(".md"):
            if self.vault_root is None:
                return None
            candidate = target if target.endswith(".md") else f"{target}.md"
            abs_candidate = str((self.vault_root / candidate).resolve())
            if abs_candidate in self._nodes:
                return abs_candidate
            return None

        candidates = self.wikilink_candidates(target)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            self.logger.warning(
                f"Wikilink [[{target}]] is ambiguous, candidates: {candidates}",
            )
        return None

    def resolve_wikilinks(self, wikilinks: list[str]) -> tuple[list[str], list[str]]:
        """Return (resolved_paths, dangling_targets)."""
        resolved: list[str] = []
        dangling: list[str] = []
        for link in wikilinks:
            hit = self.resolve_wikilink(link)
            if hit is None:
                dangling.append(link)
            else:
                resolved.append(hit)
        return resolved, dangling

    def wikilink_candidates(self, target: str) -> list[str]:
        """Paths `[[target]]` would resolve to (folder-note hit wins)."""
        folder_hits = self._folder_notes(target)
        if folder_hits:
            return folder_hits
        return self.get_paths_by_stem(target)

    def collisions_after_create(self, proposed_path: str | Path) -> list[str]:
        """Existing paths that would conflict with adding `proposed_path`."""
        p = Path(proposed_path)
        stem = p.stem
        proposed_abs = str(p.resolve())
        is_folder_note = p.parent.name == stem

        existing_folder_notes = [path for path in self._folder_notes(stem) if path != proposed_abs]
        existing_stems = [path for path in self.get_paths_by_stem(stem) if path != proposed_abs]

        if is_folder_note:
            return existing_folder_notes
        return existing_folder_notes + [sp for sp in existing_stems if sp not in existing_folder_notes]

    def all_ambiguous_wikilinks(self) -> dict[str, list[str]]:
        """Stems whose `[[stem]]` form has >1 candidate."""
        out: dict[str, list[str]] = {}
        for stem in self._stems:
            cands = self.wikilink_candidates(stem)
            if len(cands) > 1:
                out[stem] = cands
        return out

    def _folder_notes(self, stem: str) -> list[str]:
        return sorted(p for p in self._stems.get(stem, ()) if Path(p).parent.name == stem)

    # -- Graph-walk retrieval primitives ------------------------------------

    def expand_neighbors(
        self,
        seed_paths: Iterable[str],
        depth: int = 1,
        direction: str = "both",
        per_node_cap: int | None = 50,
    ) -> dict[str, int]:
        """BFS expansion over wikilink edges. Returns `{path: hop_distance}`."""
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be one of out/in/both, got {direction!r}")
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")

        seen: dict[str, int] = {}
        frontier: deque[tuple[str, int]] = deque()
        for path in seed_paths:
            if path in self._nodes and path not in seen:
                seen[path] = 0
                frontier.append((path, 0))

        while frontier:
            path, dist = frontier.popleft()
            if dist >= depth:
                continue

            neighbors: list[str] = []
            if direction in ("out", "both"):
                neighbors.extend(m.path for m, _ in self.get_links(path))
            if direction in ("in", "both"):
                neighbors.extend(m.path for m, _ in self.get_backlinks(path))

            if per_node_cap is not None and len(neighbors) > per_node_cap:
                neighbors = neighbors[:per_node_cap]

            for nb in neighbors:
                if nb in seen:
                    continue
                seen[nb] = dist + 1
                frontier.append((nb, dist + 1))

        return seen

    def subgraph_score(
        self,
        seed_paths: Iterable[str],
        decay: float = 0.5,
        depth: int = 1,
        direction: str = "both",
        per_node_cap: int | None = 50,
    ) -> dict[str, float]:
        """Decayed score per path: seed=1.0, 1-hop=decay, 2-hop=decay²..."""
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        hops = self.expand_neighbors(seed_paths, depth, direction, per_node_cap)
        return {path: decay ** hop for path, hop in hops.items()}

    def extract_anchor_paths(self, text: str) -> list[str]:
        """Pull `[[X]]` anchors from `text` and resolve each (deduped)."""
        seen: set[str] = set()
        out: list[str] = []
        for raw in extract_wikilinks(text):
            hit = self.resolve_wikilink(raw)
            if hit is not None and hit not in seen:
                seen.add(hit)
                out.append(hit)
        return out

    # -- Filter compilation -------------------------------------------------

    def filter(
        self,
        paths: list[str] | None = None,
        tags: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> ChunkFilter:
        """Compile user-facing intent into a path-set filter for chunk search."""
        cf = ChunkFilter(paths=paths, tags=tags, exclude_paths=exclude_paths)
        if cf.is_empty():
            return cf
        cf.resolved_paths = {p for p, m in self._nodes.items() if cf.match_metadata(p, m.metadata)}
        return cf

    # -- Keyword scoring utility --------------------------------------------

    @staticmethod
    def _score_keyword_match(query: str, text: str) -> float:
        words = query.split()
        if not words:
            return 0.0
        query_lower = query.lower()
        words_lower = [w.lower() for w in words]
        text_lower = text.lower()
        n_words = len(words)

        match_count = sum(1 for w in words_lower if w in text_lower)
        if match_count == 0:
            return 0.0

        base_score = match_count / n_words
        phrase_bonus = 0.2 if n_words > 1 and query_lower in text_lower else 0.0
        return min(1.0, base_score + phrase_bonus)

    # -- Filter utility (chunk side) ----------------------------------------

    @staticmethod
    def _apply_filter(chunks: list[FileChunk], chunk_filter: ChunkFilter | None) -> list[FileChunk]:
        if chunk_filter is None or chunk_filter.resolved_paths is None:
            return chunks
        return [c for c in chunks if chunk_filter.match_path(c.path)]

    # -- Default file-meta persistence (sidecar JSONL) ----------------------

    @property
    def _file_metas_path(self) -> Path:
        return self.db_path / f"{self.store_name}_files.jsonl"

    async def _persist_upsert_meta(self, meta: FileMetadata) -> None:
        """Default: full rewrite of the sidecar jsonl. Backends with proper
        relational storage (sqlite) override this for per-row UPSERT."""
        # Snapshot current in-memory state with the new entry merged in.
        # Note: in-memory not yet updated by caller — we add `meta` ourselves.
        snapshot = {**self._nodes, meta.path: meta}
        self._write_metas_jsonl(snapshot.values())

    async def _persist_delete_meta(self, path: str) -> None:
        """Default: full rewrite of the sidecar jsonl, omitting `path`."""
        snapshot = {p: m for p, m in self._nodes.items() if p != path}
        self._write_metas_jsonl(snapshot.values())

    def _iter_persisted_metas(self) -> Iterable[FileMetadata]:
        """Default: read all rows from the sidecar jsonl."""
        path = self._file_metas_path
        if not path.exists():
            return []
        out: list[FileMetadata] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(FileMetadata.model_validate_json(line))
            except Exception as e:
                self.logger.warning(f"Bad row in {path}: {e}")
        return out

    def _write_metas_jsonl(self, metas: Iterable[FileMetadata]) -> None:
        path = self._file_metas_path
        lines = [m.model_dump_json() for m in metas]
        content = "\n".join(lines)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        except Exception as e:
            self.logger.error(f"Failed to write {path}: {e}")
            raise
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- Default edge persistence (sidecar JSONL) ---------------------------

    @property
    def _edges_path(self) -> Path:
        return self.db_path / f"{self.store_name}_edges.jsonl"

    async def _persist_upsert_edges(self, path: str, edges: list[FileEdge]) -> None:
        """Default: full rewrite of the edge sidecar with `path`'s row replaced."""
        snapshot = dict(self._edges)
        if edges:
            snapshot[path] = list(edges)
        else:
            snapshot.pop(path, None)
        self._write_edges_jsonl(snapshot)

    async def _persist_delete_edges(self, path: str) -> None:
        """Default: full rewrite of the edge sidecar omitting `path`."""
        snapshot = {p: e for p, e in self._edges.items() if p != path}
        self._write_edges_jsonl(snapshot)

    def _iter_persisted_edges(self) -> Iterable[tuple[str, list[FileEdge]]]:
        """Default: yield (path, edges) rows from the sidecar jsonl."""
        path = self._edges_path
        if not path.exists():
            return []
        out: list[tuple[str, list[FileEdge]]] = []
        import json as _json
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = _json.loads(line)
                edges = [FileEdge.model_validate(e) for e in row.get("edges", [])]
                out.append((row["path"], edges))
            except Exception as e:
                self.logger.warning(f"Bad edge row in {path}: {e}")
        return out

    def _write_edges_jsonl(self, edges_by_path: dict[str, list[FileEdge]]) -> None:
        import json as _json
        path = self._edges_path
        lines = [
            _json.dumps(
                {"path": p, "edges": [e.model_dump(exclude_none=True) for e in edges]},
                ensure_ascii=False,
            )
            for p, edges in edges_by_path.items()
        ]
        content = "\n".join(lines)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        except Exception as e:
            self.logger.error(f"Failed to write {path}: {e}")
            raise
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- Abstract chunk APIs ------------------------------------------------

    @abstractmethod
    async def clear_all(self):
        """Clear all indexed data (chunks + file metas)."""

    @abstractmethod
    async def upsert_chunks(self, path: str, chunks: list[FileChunk]):
        """Insert or update all chunks for a file path. Embeddings pre-attached."""

    @abstractmethod
    async def delete_chunks(self, path: str):
        """Delete all chunks for a file path."""

    @abstractmethod
    async def get_chunks(self, path: str) -> list[FileChunk]:
        """All chunks for a file path."""

    @abstractmethod
    async def get_chunks_by_paths(self, paths: Iterable[str]) -> list[FileChunk]:
        """Batch fetch chunks across many paths (used by graph-walk retrieval)."""

    @abstractmethod
    async def vector_search(
        self,
        query: str,
        limit: int,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        """Vector similarity search."""

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        limit: int,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        """Full-text/keyword search."""
