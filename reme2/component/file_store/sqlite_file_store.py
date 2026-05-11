"""SQLite file store backend (chunks + nodes).

File nodes live in `nodes_{store_name}` (path PK, st_mtime, data JSON
where `data` carries everything else — frontmatter extras + edges).
Chunks live in `chunks_{store_name}` with companion `vec0` and `fts5`
virtual tables for vector and trigram search.
"""

import json
import sqlite3
import struct
import time
from typing import Iterable

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import ChunkFilter, FileChunk, FileNode
from ...utils.chunk_search import filter_chunks, keyword_score


@R.register("sqlite")
class SqliteFileStore(BaseFileStore):
    """SQLite file store with vector + full-text search.

    Uses sqlite-vec for vector similarity search and FTS5 with trigram
    tokenizer for keyword search. Nodes (frontmatter + edges + mtime)
    live in a relational table with native UPSERT — overrides the
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
    def nodes_table(self) -> str:
        return f"nodes_{self.store_name}"

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

    @property
    def _embedding_dim(self) -> int:
        return self.embedding_model.dimensions if self.embedding_model else 1024

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
                        embedding FLOAT[{self._embedding_dim}]
                    )
                """,
                )
                self.logger.info(f"Created vector table (dims={self._embedding_dim})")

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
                CREATE TABLE IF NOT EXISTS {self.nodes_table} (
                    path TEXT PRIMARY KEY,
                    st_mtime REAL,
                    data TEXT
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

    # -- Search operations --------------------------------------------------

    async def vector_search(
        self,
        query: str,
        limit: int,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        if not self.vector_enabled or not query:
            return []

        embeddings = await self.embed([query])
        query_embedding = embeddings[0] if embeddings else None
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

            chunks = filter_chunks(chunks, chunk_filter)
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
                return filter_chunks(results, chunk_filter)[:limit]

        return filter_chunks(
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
                score = keyword_score(phrase, text)
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
            cursor.execute(f"DELETE FROM {self.nodes_table}")
            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.error(f"Failed to clear all data: {e}")
            raise
        finally:
            cursor.close()
        self.logger.info(f"Cleared all data from SqliteFileStore '{self.store_name}'")

    # -- Node persistence (overrides default JSONL sidecar) -----------------

    async def _persist_upsert_node(self, node: FileNode) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"""INSERT INTO {self.nodes_table}
                    (path, st_mtime, data)
                    VALUES (?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        st_mtime = excluded.st_mtime,
                        data = excluded.data""",
                (node.path, node.st_mtime, node.model_dump_json()),
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to upsert node {node.path}: {e}")
            raise
        finally:
            cursor.close()

    async def _persist_delete_node(self, path: str) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"DELETE FROM {self.nodes_table} WHERE path = ?", (path,))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to delete node {path}: {e}")
            raise
        finally:
            cursor.close()

    def _iter_persisted_nodes(self) -> Iterable[FileNode]:
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT data FROM {self.nodes_table}")
            out: list[FileNode] = []
            for (data_json,) in cursor.fetchall():
                if not data_json:
                    continue
                try:
                    out.append(FileNode.model_validate_json(data_json))
                except Exception as e:
                    self.logger.warning(f"Bad node row: {e}")
            return out
        finally:
            cursor.close()
