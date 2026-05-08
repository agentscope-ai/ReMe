"""SQLite file store backend (chunks + file metadata)."""

import json
import sqlite3
import struct
import time
from typing import Iterable

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import ChunkFilter, FileChunk, FileEdge, FileMetadata


@R.register("sqlite")
class SqliteFileStore(BaseFileStore):
    """SQLite file store with vector + full-text search.

    Uses sqlite-vec for vector similarity search and FTS5 with trigram
    tokenizer for keyword search. File metadata lives in a relational
    table (`files_{store_name}`) with native UPSERT — overrides the
    base class's default JSONL sidecar.
    """

    def __init__(self, vec_ext_path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.vec_ext_path = vec_ext_path
        self.conn: sqlite3.Connection | None = None

    # -- Table names --------------------------------------------------------

    @property
    def chunks_table(self) -> str:
        return f"chunks_{self.store_name}"

    @property
    def vector_table(self) -> str:
        return f"chunks_vec_{self.store_name}"

    @property
    def fts_table(self) -> str:
        return f"chunks_fts_{self.store_name}"

    @property
    def files_table(self) -> str:
        return f"files_{self.store_name}"

    @property
    def edges_table(self) -> str:
        return f"edges_{self.store_name}"

    @staticmethod
    def vector_to_blob(embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self) -> None:
        self.conn = sqlite3.connect(self.db_path / "reme.db", check_same_thread=False)

        if self.vector_enabled:
            self.conn.enable_load_extension(True)
            if self.vec_ext_path:
                try:
                    self.conn.load_extension(self.vec_ext_path)
                    self.logger.info(f"Loaded sqlite-vec: {self.vec_ext_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load sqlite-vec: {e}")
                    self._disable_vector_search(f"extension load failed: {e}")
            else:
                loaded = False
                try:
                    import sqlite_vec

                    ext_path = sqlite_vec.loadable_path()
                    self.conn.load_extension(ext_path)
                    self.logger.info(f"Loaded sqlite-vec from package: {ext_path}")
                    loaded = True
                except Exception:
                    pass

                if not loaded:
                    for name in ["vec0", "sqlite_vec", "vector0"]:
                        try:
                            self.conn.load_extension(name)
                            self.logger.info(f"Loaded sqlite-vec: {name}")
                            loaded = True
                            break
                        except Exception:
                            pass

                if not loaded:
                    self._disable_vector_search("sqlite-vec extension not available")

            self.conn.enable_load_extension(False)

        await self._create_tables()
        self.logger.info(f"SqliteFileStore '{self.store_name}' ready: db={self.db_path / 'reme.db'}")
        await super()._start()

    async def _create_tables(self) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                    id TEXT PRIMARY KEY,
                    path TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    hash TEXT,
                    text TEXT,
                    embedding TEXT,
                    updated_at INTEGER
                )
            """,
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.chunks_table}_path " f"ON {self.chunks_table}(path)",
            )

            if self.vector_enabled:
                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.vector_table} USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{self.embedding_dim}]
                    )
                """,
                )
                self.logger.info(f"Created vector table (dims={self.embedding_dim})")

            if self.fts_enabled:
                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table} USING fts5(
                        text,
                        id UNINDEXED,
                        path UNINDEXED,
                        start_line UNINDEXED,
                        end_line UNINDEXED,
                        tokenize='trigram'
                    )
                """,
                )
                self.logger.info("Created FTS5 table with trigram tokenizer")

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.files_table} (
                    path TEXT PRIMARY KEY,
                    file TEXT,
                    st_mtime REAL,
                    metadata TEXT
                )
            """,
            )

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.edges_table} (
                    path TEXT PRIMARY KEY,
                    edges TEXT
                )
            """,
            )

            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
        finally:
            cursor.close()

    async def _close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        await super()._close()

    # -- Write operations ---------------------------------------------------

    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            old_ids = [
                row[0]
                for row in cursor.execute(
                    f"SELECT id FROM {self.chunks_table} WHERE path = ?",
                    (path,),
                ).fetchall()
            ]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                cursor.execute(f"DELETE FROM {self.chunks_table} WHERE id IN ({placeholders})", old_ids)
                if self.vector_enabled:
                    for oid in old_ids:
                        cursor.execute(f"DELETE FROM {self.vector_table} WHERE id = ?", (oid,))
                if self.fts_enabled:
                    cursor.execute(f"DELETE FROM {self.fts_table} WHERE path = ?", (path,))

            if chunks:
                now = int(time.time() * 1000)
                for chunk in chunks:
                    cursor.execute(
                        f"""INSERT INTO {self.chunks_table}
                            (id, path, start_line, end_line, hash, text, embedding, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk.id,
                            path,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.hash,
                            chunk.text,
                            json.dumps(chunk.embedding) if chunk.embedding else None,
                            now,
                        ),
                    )

                    if self.vector_enabled and chunk.embedding:
                        cursor.execute(
                            f"INSERT INTO {self.vector_table} (id, embedding) VALUES (?, ?)",
                            (chunk.id, self.vector_to_blob(chunk.embedding)),
                        )

                    if self.fts_enabled:
                        cursor.execute(
                            f"""INSERT INTO {self.fts_table}
                                (text, id, path, start_line, end_line)
                                VALUES (?, ?, ?, ?, ?)""",
                            (chunk.text, chunk.id, path, chunk.start_line, chunk.end_line),
                        )

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.error(f"Failed to upsert chunks for {path}: {e}")
            raise
        finally:
            cursor.close()

    async def delete_chunks(self, path: str) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            chunk_ids = [
                row[0]
                for row in cursor.execute(
                    f"SELECT id FROM {self.chunks_table} WHERE path = ?",
                    (path,),
                ).fetchall()
            ]

            if self.vector_enabled and chunk_ids:
                for cid in chunk_ids:
                    cursor.execute(f"DELETE FROM {self.vector_table} WHERE id = ?", (cid,))

            if self.fts_enabled:
                cursor.execute(f"DELETE FROM {self.fts_table} WHERE path = ?", (path,))

            cursor.execute(f"DELETE FROM {self.chunks_table} WHERE path = ?", (path,))

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.error(f"Failed to delete chunks for {path}: {e}")
            raise
        finally:
            cursor.close()

    # -- Read operations ----------------------------------------------------

    async def get_chunks(self, path: str) -> list[FileChunk]:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"""SELECT id, path, start_line, end_line, text, hash, embedding
                    FROM {self.chunks_table} WHERE path = ?
                    ORDER BY start_line""",
                (path,),
            )
            chunks = []
            for row in cursor.fetchall():
                chunk_id, path_val, start, end, text, hash_val, emb_str = row
                embedding = None
                if emb_str:
                    try:
                        embedding = json.loads(emb_str)
                    except (json.JSONDecodeError, TypeError):
                        pass
                chunks.append(
                    FileChunk(
                        id=chunk_id,
                        path=path_val,
                        start_line=start,
                        end_line=end,
                        text=text,
                        hash=hash_val,
                        embedding=embedding,
                    ),
                )
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to get chunks for {path}: {e}")
            return []
        finally:
            cursor.close()

    async def get_chunks_by_paths(self, paths: Iterable[str]) -> list[FileChunk]:
        """Batch fetch chunks across many paths.

        Splits into ≤900-placeholder batches to stay under sqlite's
        SQLITE_MAX_VARIABLE_NUMBER (default 999, leave headroom).
        """
        wanted = list({p for p in paths})
        if not wanted:
            return []

        cursor = self.conn.cursor()
        try:
            chunks: list[FileChunk] = []
            BATCH = 900
            for offset in range(0, len(wanted), BATCH):
                batch = wanted[offset : offset + BATCH]
                placeholders = ",".join("?" * len(batch))
                cursor.execute(
                    f"""SELECT id, path, start_line, end_line, text, hash, embedding
                        FROM {self.chunks_table} WHERE path IN ({placeholders})
                        ORDER BY path, start_line""",
                    batch,
                )
                for row in cursor.fetchall():
                    chunk_id, path_val, start_line, end, text, hash_val, emb_str = row
                    embedding = None
                    if emb_str:
                        try:
                            embedding = json.loads(emb_str)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    chunks.append(
                        FileChunk(
                            id=chunk_id,
                            path=path_val,
                            start_line=start_line,
                            end_line=end,
                            text=text,
                            hash=hash_val,
                            embedding=embedding,
                        ),
                    )
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to batch get chunks ({len(wanted)} paths): {e}")
            return []
        finally:
            cursor.close()

    # -- Search helpers -----------------------------------------------------

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        if not query:
            return ""
        special_chars = list("*?:^()[]{}'\"`|+-=<>!@#$%&\\/,;")
        cleaned = query
        for ch in special_chars:
            cleaned = cleaned.replace(ch, " ")
        return " ".join(cleaned.split())

    @staticmethod
    def _path_filter_clause(chunk_filter: ChunkFilter | None, column: str = "path") -> tuple[str, list]:
        """Build a WHERE clause fragment for ChunkFilter; returns (sql_fragment, params)."""
        if chunk_filter is None or chunk_filter.resolved_paths is None:
            return "", []
        paths = chunk_filter.resolved_paths
        if not paths:
            return "0", []  # filter excludes everything
        placeholders = ",".join("?" * len(paths))
        return f"{column} IN ({placeholders})", list(paths)

    # -- Search operations --------------------------------------------------

    async def vector_search(
        self,
        query: str,
        limit: int,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        if not self.vector_enabled or not query:
            return []

        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        cursor = self.conn.cursor()
        try:
            query_blob = self.vector_to_blob(query_embedding)
            cursor.execute(
                f"""
                SELECT c.id, c.path, c.start_line, c.end_line, c.text, v.distance
                FROM {self.vector_table} v
                JOIN {self.chunks_table} c ON v.id = c.id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                [query_blob, limit],
            )

            chunks = []
            for cid, path, start, end, text, dist in cursor.fetchall():
                score = max(0.0, 1.0 - dist / 2.0)
                chunks.append(
                    FileChunk(
                        id=cid,
                        path=path,
                        start_line=start,
                        end_line=end,
                        text=text,
                        hash="",
                        scores={"vector": score, "score": score},
                    ),
                )

            chunks = self._apply_filter(chunks, chunk_filter)
            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[:limit]
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
        finally:
            cursor.close()

    async def keyword_search(
        self,
        query: str,
        limit: int,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        if not self.fts_enabled or not query:
            return []

        cleaned = self._sanitize_fts_query(query)
        if not cleaned:
            return []

        words = cleaned.split()
        if not words:
            return []

        if all(len(w) >= 3 for w in words):
            results = await self._fts_trigram_search(words, limit)
            if results:
                return self._apply_filter(results, chunk_filter)[:limit]

        return self._apply_filter(
            await self._like_search(cleaned, words, limit),
            chunk_filter,
        )[:limit]

    async def _fts_trigram_search(self, words: list[str], limit: int) -> list[FileChunk]:
        escaped = [w.replace('"', '""') for w in words]
        fts_query = " OR ".join(escaped)

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"""
                SELECT fts.id, fts.path, fts.start_line, fts.end_line, fts.text, rank
                FROM {self.fts_table} fts
                WHERE fts.text MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                [fts_query, limit],
            )

            chunks = []
            for cid, path, start, end, text, rank in cursor.fetchall():
                score = max(0.0, 1.0 / (1.0 + abs(rank)))
                chunks.append(
                    FileChunk(
                        id=cid,
                        path=path,
                        start_line=start,
                        end_line=end,
                        text=text,
                        hash="",
                        scores={"keyword": score, "score": score},
                    ),
                )
            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks
        except Exception as e:
            self.logger.error(f"FTS trigram search failed: {e}")
            return []
        finally:
            cursor.close()

    async def _like_search(self, phrase: str, words: list[str], limit: int) -> list[FileChunk]:
        cursor = self.conn.cursor()
        try:
            like_clauses = []
            params: list = []
            for word in words:
                like_clauses.append("text LIKE ?")
                params.append(f"%{word}%")

            where_clause = " OR ".join(like_clauses)
            fetch_limit = min(limit * 3, 200)
            params.append(fetch_limit)

            cursor.execute(
                f"""
                SELECT id, path, start_line, end_line, text
                FROM {self.chunks_table}
                WHERE ({where_clause})
                LIMIT ?
                """,
                params,
            )

            chunks = []
            for cid, path, start, end, text in cursor.fetchall():
                score = self._score_keyword_match(phrase, text)
                if score == 0.0:
                    continue

                chunks.append(
                    FileChunk(
                        id=cid,
                        path=path,
                        start_line=start,
                        end_line=end,
                        text=text,
                        hash="",
                        scores={"keyword": score, "score": score},
                    ),
                )

            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[:limit]
        except Exception as e:
            self.logger.error(f"LIKE search failed: {e}")
            return []
        finally:
            cursor.close()

    # -- Clear --------------------------------------------------------------

    async def clear_all(self) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute(f"DELETE FROM {self.chunks_table}")
            if self.vector_enabled:
                cursor.execute(f"DELETE FROM {self.vector_table}")
            if self.fts_enabled:
                cursor.execute(f"DELETE FROM {self.fts_table}")
            cursor.execute(f"DELETE FROM {self.files_table}")
            cursor.execute(f"DELETE FROM {self.edges_table}")
            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.error(f"Failed to clear all data: {e}")
            raise
        finally:
            cursor.close()
        self.logger.info(f"Cleared all data from SqliteFileStore '{self.store_name}'")

    # -- File-meta persistence (overrides default JSONL sidecar) ------------

    async def _persist_upsert_meta(self, meta: FileMetadata) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"""INSERT INTO {self.files_table}
                    (path, file, st_mtime, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        file = excluded.file,
                        st_mtime = excluded.st_mtime,
                        metadata = excluded.metadata""",
                (
                    meta.path,
                    meta.file,
                    meta.st_mtime,
                    json.dumps(meta.metadata, ensure_ascii=False, default=str),
                ),
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to upsert file meta {meta.path}: {e}")
            raise
        finally:
            cursor.close()

    async def _persist_delete_meta(self, path: str) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"DELETE FROM {self.files_table} WHERE path = ?", (path,))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to delete file meta {path}: {e}")
            raise
        finally:
            cursor.close()

    def _iter_persisted_metas(self) -> Iterable[FileMetadata]:
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"SELECT path, file, st_mtime, metadata "
                f"FROM {self.files_table}",
            )
            out: list[FileMetadata] = []
            for path, file, st_mtime, metadata_json in cursor.fetchall():
                try:
                    out.append(
                        FileMetadata(
                            path=path,
                            file=file or "",
                            st_mtime=st_mtime or 0.0,
                            metadata=json.loads(metadata_json) if metadata_json else {},
                        ),
                    )
                except Exception as e:
                    self.logger.warning(f"Bad file meta row for {path}: {e}")
            return out
        finally:
            cursor.close()

    # -- Edge persistence (overrides default JSONL sidecar) -----------------

    async def _persist_upsert_edges(self, path: str, edges: list[FileEdge]) -> None:
        cursor = self.conn.cursor()
        try:
            if edges:
                cursor.execute(
                    f"""INSERT INTO {self.edges_table} (path, edges)
                        VALUES (?, ?)
                        ON CONFLICT(path) DO UPDATE SET edges = excluded.edges""",
                    (path, json.dumps(
                        [e.model_dump(exclude_none=True) for e in edges],
                        ensure_ascii=False,
                    )),
                )
            else:
                cursor.execute(f"DELETE FROM {self.edges_table} WHERE path = ?", (path,))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to upsert edges for {path}: {e}")
            raise
        finally:
            cursor.close()

    async def _persist_delete_edges(self, path: str) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"DELETE FROM {self.edges_table} WHERE path = ?", (path,))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to delete edges for {path}: {e}")
            raise
        finally:
            cursor.close()

    def _iter_persisted_edges(self) -> Iterable[tuple[str, list[FileEdge]]]:
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT path, edges FROM {self.edges_table}")
            out: list[tuple[str, list[FileEdge]]] = []
            for path, edges_json in cursor.fetchall():
                try:
                    raws = json.loads(edges_json) if edges_json else []
                    out.append((path, [FileEdge.model_validate(e) for e in raws]))
                except Exception as e:
                    self.logger.warning(f"Bad edge row for {path}: {e}")
            return out
        finally:
            cursor.close()
