"""SQLite storage backend for file store."""

import json
import sqlite3
import struct
import time

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import FileChunk, FileMetadata, SearchFilter


@R.register("sqlite")
class SqliteFileStore(BaseFileStore):
    """SQLite file storage with vector and full-text search.

    Uses sqlite-vec for vector similarity search and FTS5 with trigram
    tokenizer for keyword search. Falls back to LIKE-based substring
    search for short query terms.
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
    def files_table(self) -> str:
        return f"files_{self.store_name}"

    @property
    def vector_table(self) -> str:
        return f"chunks_vec_{self.store_name}"

    @property
    def fts_table(self) -> str:
        return f"chunks_fts_{self.store_name}"

    @staticmethod
    def vector_to_blob(embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self, app_context=None) -> None:
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
        self.logger.info(
            f"SqliteFileStore '{self.store_name}' ready: "
            f"db={self.db_path / 'reme.db'}",
        )
        await super()._start(app_context)

    async def _create_tables(self) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.files_table} (
                    path TEXT PRIMARY KEY,
                    hash TEXT,
                    mtime REAL,
                    size INTEGER,
                    metadata TEXT,
                    chunk_count INTEGER DEFAULT 0
                )
            """)

            cursor.execute(f"""
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
            """)

            if self.vector_enabled:
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.vector_table} USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{self.embedding_dim}]
                    )
                """)
                self.logger.info(f"Created vector table (dims={self.embedding_dim})")

            if self.fts_enabled:
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table} USING fts5(
                        text,
                        id UNINDEXED,
                        path UNINDEXED,
                        start_line UNINDEXED,
                        end_line UNINDEXED,
                        tokenize='trigram'
                    )
                """)
                self.logger.info("Created FTS5 table with trigram tokenizer")

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

    # -- Metadata helpers ---------------------------------------------------

    async def _get_all_metadata(self) -> dict[str, FileMetadata]:
        """Load all file metadata from SQL into a dict for tag filtering."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT path, hash, mtime, size, metadata FROM {self.files_table}")
            result = {}
            for path, hash_val, mtime, size, meta_str in cursor.fetchall():
                metadata = json.loads(meta_str) if meta_str else {}
                result[path] = FileMetadata(
                    hash=hash_val,
                    mtime_ms=mtime,
                    size=size,
                    path=path,
                    metadata=metadata,
                )
            return result
        except Exception as e:
            self.logger.error(f"Failed to load all metadata: {e}")
            return {}
        finally:
            cursor.close()

    # -- Write operations ---------------------------------------------------

    async def upsert_file(self, file_meta: FileMetadata, chunks: list[FileChunk]) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            # Upsert file metadata
            cursor.execute(
                f"""INSERT OR REPLACE INTO {self.files_table}
                    (path, hash, mtime, size, metadata, chunk_count)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    file_meta.path,
                    file_meta.hash,
                    file_meta.mtime_ms,
                    file_meta.size,
                    json.dumps(file_meta.metadata, ensure_ascii=False) if file_meta.metadata else None,
                    len(chunks),
                ),
            )

            # Delete old chunks and vectors/fts for this file
            old_ids = [
                row[0] for row in cursor.execute(
                    f"SELECT id FROM {self.chunks_table} WHERE path = ?",
                    (file_meta.path,),
                ).fetchall()
            ]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                cursor.execute(f"DELETE FROM {self.chunks_table} WHERE id IN ({placeholders})", old_ids)
                if self.vector_enabled:
                    for oid in old_ids:
                        cursor.execute(f"DELETE FROM {self.vector_table} WHERE id = ?", (oid,))
                if self.fts_enabled:
                    cursor.execute(f"DELETE FROM {self.fts_table} WHERE path = ?", (file_meta.path,))

            # Insert new chunks
            if chunks:
                chunks = await self.get_chunk_embeddings(chunks)
                now = int(time.time() * 1000)
                for chunk in chunks:
                    cursor.execute(
                        f"""INSERT INTO {self.chunks_table}
                            (id, path, start_line, end_line, hash, text, embedding, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk.id,
                            file_meta.path,
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
                            (chunk.text, chunk.id, file_meta.path, chunk.start_line, chunk.end_line),
                        )

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.error(f"Failed to upsert file {file_meta.path}: {e}")
            raise
        finally:
            cursor.close()

    async def delete_file(self, path: str) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            chunk_ids = [
                row[0] for row in cursor.execute(
                    f"SELECT id FROM {self.chunks_table} WHERE path = ?", (path,),
                ).fetchall()
            ]

            if self.vector_enabled and chunk_ids:
                for cid in chunk_ids:
                    cursor.execute(f"DELETE FROM {self.vector_table} WHERE id = ?", (cid,))

            if self.fts_enabled:
                cursor.execute(f"DELETE FROM {self.fts_table} WHERE path = ?", (path,))

            cursor.execute(f"DELETE FROM {self.chunks_table} WHERE path = ?", (path,))
            cursor.execute(f"DELETE FROM {self.files_table} WHERE path = ?", (path,))

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.error(f"Failed to delete file {path}: {e}")
            raise
        finally:
            cursor.close()

    # -- Read operations (SQL overrides) ------------------------------------

    async def list_files(self) -> list[str]:
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT path FROM {self.files_table}")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            return []
        finally:
            cursor.close()

    async def get_file_metadata(self, path: str) -> FileMetadata | None:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"SELECT hash, mtime, size, metadata, chunk_count FROM {self.files_table} WHERE path = ?",
                (path,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            hash_val, mtime, size, meta_str, chunk_count = row
            metadata = json.loads(meta_str) if meta_str else {}
            return FileMetadata(
                hash=hash_val,
                mtime_ms=mtime,
                size=size,
                path=path,
                chunk_count=chunk_count,
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"Failed to get file metadata for {path}: {e}")
            return None
        finally:
            cursor.close()

    async def get_file_chunks(self, path: str) -> list[FileChunk]:
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
                chunks.append(FileChunk(
                    id=chunk_id,
                    path=path_val,
                    start_line=start,
                    end_line=end,
                    text=text,
                    hash=hash_val,
                    embedding=embedding,
                ))
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to get file chunks for {path}: {e}")
            return []
        finally:
            cursor.close()

    # -- Search helpers -----------------------------------------------------

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        if not query:
            return ""
        special_chars = list('*?:^()[]{}\'"`|+-=<>!@#$%&\\/,;')
        cleaned = query
        for ch in special_chars:
            cleaned = cleaned.replace(ch, " ")
        return " ".join(cleaned.split())

    # -- Search operations --------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: SearchFilter | None = None) -> list[FileChunk]:
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
                chunk = FileChunk(
                    id=cid,
                    path=path,
                    start_line=start,
                    end_line=end,
                    text=text,
                    hash="",
                    scores={"vector": score, "score": score},
                )
                chunks.append(chunk)

            # Apply filter with SQL-based metadata for tag support
            file_meta = await self._get_all_metadata() if (search_filter and search_filter.tags) else None
            chunks = self._apply_filter(chunks, search_filter, file_meta)
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
            search_filter: SearchFilter | None = None,
    ) -> list[FileChunk]:
        if not self.fts_enabled or not query:
            return []

        cleaned = self._sanitize_fts_query(query)
        if not cleaned:
            return []

        words = cleaned.split()
        if not words:
            return []

        file_meta = await self._get_all_metadata() if (search_filter and search_filter.tags) else None

        # FTS5 trigram requires all terms >= 3 chars
        if all(len(w) >= 3 for w in words):
            results = await self._fts_trigram_search(words, limit)
            if results:
                return self._apply_filter(results, search_filter, file_meta)[:limit]

        return self._apply_filter(
            await self._like_search(cleaned, words, limit),
            search_filter,
            file_meta,
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
                chunk = FileChunk(
                    id=cid,
                    path=path,
                    start_line=start,
                    end_line=end,
                    text=text,
                    hash="",
                    scores={"keyword": score, "score": score},
                )
                chunks.append(chunk)
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

                chunk = FileChunk(
                    id=cid,
                    path=path,
                    start_line=start,
                    end_line=end,
                    text=text,
                    hash="",
                    scores={"keyword": score, "score": score},
                )
                chunks.append(chunk)

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
            cursor.execute(f"DELETE FROM {self.files_table}")
            cursor.execute(f"DELETE FROM {self.chunks_table}")
            if self.vector_enabled:
                cursor.execute(f"DELETE FROM {self.vector_table}")
            if self.fts_enabled:
                cursor.execute(f"DELETE FROM {self.fts_table}")
            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.error(f"Failed to clear all data: {e}")
            raise
        finally:
            cursor.close()
        self.logger.info(f"Cleared all data from SqliteFileStore '{self.store_name}'")
