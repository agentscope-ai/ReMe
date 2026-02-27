"""SQLite storage backend for file store."""

import json
import sqlite3
import struct
import time

from loguru import logger

from .base_file_store import BaseFileStore
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk, MemorySearchResult


class SqliteFileStore(BaseFileStore):
    """SQLite file storage with vector and full-text search.

    Inherits embedding methods from BaseFileStore:
    - get_chunk_embedding / get_chunk_embeddings (async)
    - get_chunk_embedding_sync / get_chunk_embeddings_sync (sync)
    - get_embedding / get_embeddings (async)

    Provides SQLite-backed persistent storage with:
    - Vector similarity search (via sqlite-vec extension)
    - Full-text search (via FTS5)
    - Efficient chunk and file metadata management
    """

    def __init__(self, vec_ext_path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.vec_ext_path = vec_ext_path

        self.conn: sqlite3.Connection | None = None

    @property
    def vector_table_name(self) -> str:
        """Get the name of the vector table for this store."""
        return f"chunks_vec_{self.store_name}"

    @property
    def fts_table_name(self) -> str:
        """Get the name of the FTS table for this store."""
        return f"chunks_fts_{self.store_name}"

    @property
    def chunks_table_name(self) -> str:
        """Get the name of the chunks table for this store."""
        return f"chunks_{self.store_name}"

    @property
    def files_table_name(self) -> str:
        """Get the name of the files table for this store."""
        return f"files_{self.store_name}"

    @staticmethod
    def vector_to_blob(embedding: list[float]) -> bytes:
        """Convert vector to binary blob for sqlite-vec."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    async def start(self) -> None:
        """Initialize database and load extensions."""
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(self.db_path / "reme.db", check_same_thread=False)

        # Only load sqlite-vec extension if vector search is enabled
        if self.vector_enabled:
            logger.warning(
                "On macOS systems with version 14 or earlier, "
                "loading the sqlite-vec vector extension carries a risk of crashes or hangs.",
            )

            self.conn.enable_load_extension(True)

            # Load sqlite-vec extension
            if self.vec_ext_path:
                try:
                    self.conn.load_extension(self.vec_ext_path)
                    logger.info(f"Loaded sqlite-vec: {self.vec_ext_path}")
                except Exception as e:
                    logger.warning(f"Failed to load sqlite-vec: {e}")

            else:
                try:
                    import sqlite_vec

                    ext_path = sqlite_vec.loadable_path()
                    self.conn.load_extension(ext_path)
                    logger.info(f"Loaded sqlite-vec from package: {ext_path}")

                except Exception as e:
                    logger.warning(f"Failed to load sqlite-vec from package: {e}")
                    # Fallback: try common extension names
                    for name in ["vec0", "sqlite_vec", "vector0"]:
                        try:
                            self.conn.load_extension(name)
                            logger.info(f"Loaded sqlite-vec: {name}")
                            break
                        except Exception:
                            pass

            self.conn.enable_load_extension(False)
        else:
            logger.info("Vector search disabled, skipping sqlite-vec extension loading")

        await self._create_tables()

    async def _create_tables(self) -> None:
        """Create database schema."""
        cursor = self.conn.cursor()
        try:
            # Files
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.files_table_name} (
                    path TEXT,
                    source TEXT,
                    hash TEXT,
                    mtime REAL,
                    size INTEGER,
                    PRIMARY KEY (path, source)
                )
            """,
            )

            # Chunks
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.chunks_table_name} (
                    id TEXT PRIMARY KEY,
                    path TEXT,
                    source TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    hash TEXT,
                    text TEXT,
                    embedding TEXT,
                    updated_at INTEGER
                )
            """,
            )

            # Vector table (sqlite-vec)
            if self.vector_enabled:
                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.vector_table_name} USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{self.embedding_dim}]
                    )
                """,
                )
                logger.info(f"Created vector table (dims={self.embedding_dim})")

            # FTS table
            if self.fts_enabled:
                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table_name} USING fts5(
                        text,
                        id UNINDEXED,
                        path UNINDEXED,
                        source UNINDEXED,
                        start_line UNINDEXED,
                        end_line UNINDEXED,
                        tokenize='trigram'
                    )
                """,
                )
                logger.info("Created FTS5 table with trigram tokenizer")

            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
        finally:
            cursor.close()

    async def upsert_file(self, file_meta: FileMetadata, source: MemorySource, chunks: list[MemoryChunk]):
        """Insert or update file and its chunks."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("BEGIN")

            # Insert file
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.files_table_name} (path, source, hash, mtime, size)
                VALUES (?, ?, ?, ?, ?)
            """,
                (file_meta.path, source.value, file_meta.hash, file_meta.mtime_ms, file_meta.size),
            )

            # Insert chunks
            now = int(time.time() * 1000)
            for chunk in chunks:
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.chunks_table_name} (
                        id, path, source, start_line, end_line,
                        hash, text, embedding, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.id,
                        file_meta.path,
                        source.value,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.hash,
                        chunk.text,
                        json.dumps(chunk.embedding) if chunk.embedding else None,
                        now,
                    ),
                )

                # Insert vector (vec0 doesn't support OR REPLACE, use DELETE + INSERT)
                if self.vector_enabled:
                    if not chunk.embedding:
                        logger.warning(
                            f"Chunk {chunk.id} missing embedding for vector insert, skipping vector indexing",
                        )
                    else:
                        # Delete existing vector first
                        cursor.execute(
                            f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                            (chunk.id,),
                        )
                        # Then insert new vector
                        cursor.execute(
                            f"INSERT INTO {self.vector_table_name} (id, embedding) VALUES (?, ?)",
                            (chunk.id, self.vector_to_blob(chunk.embedding)),
                        )

                # Insert FTS
                if self.fts_enabled:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.fts_table_name} (
                            text, id, path, source, start_line, end_line
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            chunk.text,
                            chunk.id,
                            file_meta.path,
                            source.value,
                            chunk.start_line,
                            chunk.end_line,
                        ),
                    )

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Failed to upsert file {file_meta.path}: {e}")
            raise
        finally:
            cursor.close()

    async def delete_file(self, path: str, source: MemorySource):
        """Delete file and all its chunks."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            # Get chunk IDs for vector deletion
            cursor.execute(
                f"SELECT id FROM {self.chunks_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            # Delete vectors
            if self.vector_enabled and chunk_ids:
                for chunk_id in chunk_ids:
                    cursor.execute(
                        f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                        (chunk_id,),
                    )

            # Delete FTS entries
            if self.fts_enabled:
                cursor.execute(
                    f"DELETE FROM {self.fts_table_name} WHERE path = ? AND source = ?",
                    (path, source.value),
                )

            # Delete chunks and file
            cursor.execute(
                f"DELETE FROM {self.chunks_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )
            cursor.execute(
                f"DELETE FROM {self.files_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Failed to delete file {path}: {e}")
            raise
        finally:
            cursor.close()

    async def delete_file_chunks(self, path: str, chunk_ids: list[str]):
        """Delete specific chunks for a file."""
        if not chunk_ids:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            # Delete vectors
            if self.vector_enabled:
                for chunk_id in chunk_ids:
                    cursor.execute(
                        f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                        (chunk_id,),
                    )

            # Delete FTS entries
            if self.fts_enabled:
                placeholders = ",".join("?" * len(chunk_ids))
                cursor.execute(
                    f"DELETE FROM {self.fts_table_name} WHERE id IN ({placeholders})",
                    chunk_ids,
                )

            # Delete chunks
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(
                f"DELETE FROM {self.chunks_table_name} WHERE id IN ({placeholders})",
                chunk_ids,
            )

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Failed to delete chunks for {path}: {e}")
            raise
        finally:
            cursor.close()

    async def upsert_chunks(self, chunks: list[MemoryChunk], source: MemorySource):
        """Insert or update specific chunks without affecting other chunks."""
        if not chunks:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            now = int(time.time() * 1000)
            for chunk in chunks:
                # Insert/update chunk
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.chunks_table_name} (
                        id, path, source, start_line, end_line,
                        hash, text, embedding, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.id,
                        chunk.path,
                        source.value,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.hash,
                        chunk.text,
                        json.dumps(chunk.embedding) if chunk.embedding else None,
                        now,
                    ),
                )

                # Insert/update vector (vec0 doesn't support OR REPLACE, use DELETE + INSERT)
                if self.vector_enabled:
                    if not chunk.embedding:
                        logger.warning(
                            f"Chunk {chunk.id} missing embedding for vector insert, skipping vector indexing",
                        )
                    else:
                        # Delete existing vector first
                        cursor.execute(
                            f"DELETE FROM {self.vector_table_name} WHERE id = ?",
                            (chunk.id,),
                        )
                        # Then insert new vector
                        cursor.execute(
                            f"INSERT INTO {self.vector_table_name} (id, embedding) VALUES (?, ?)",
                            (chunk.id, self.vector_to_blob(chunk.embedding)),
                        )

                # Insert/update FTS
                if self.fts_enabled:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.fts_table_name} (
                            text, id, path, source, start_line, end_line
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            chunk.text,
                            chunk.id,
                            chunk.path,
                            source.value,
                            chunk.start_line,
                            chunk.end_line,
                        ),
                    )

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Failed to upsert chunks: {e}")
            raise
        finally:
            cursor.close()

    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed files."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT path FROM {self.files_table_name} WHERE source = ?", (source.value,))
            paths = [row[0] for row in cursor.fetchall()]
            return paths
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise
        finally:
            cursor.close()

    async def get_file_metadata(self, path: str, source: MemorySource) -> FileMetadata | None:
        """Get file metadata with chunk count."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"SELECT hash, mtime, size FROM {self.files_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )
            row = cursor.fetchone()
            if not row:
                return None

            hash_val, mtime, size = row
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.chunks_table_name} WHERE path = ? AND source = ?",
                (path, source.value),
            )
            chunk_count = cursor.fetchone()[0]

            return FileMetadata(
                hash=hash_val,
                mtime_ms=mtime,
                size=size,
                path=path,
                chunk_count=chunk_count,
            )
        except Exception as e:
            logger.error(f"Failed to get file metadata for {path}: {e}")
            raise
        finally:
            cursor.close()

    async def update_file_metadata(self, file_meta: FileMetadata, source: MemorySource) -> None:
        """Update file metadata without affecting chunks."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.files_table_name} (path, source, hash, mtime, size)
                VALUES (?, ?, ?, ?, ?)
            """,
                (file_meta.path, source.value, file_meta.hash, file_meta.mtime_ms, file_meta.size),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to update file metadata for {file_meta.path}: {e}")
            raise
        finally:
            cursor.close()

    async def get_file_chunks(self, path: str, source: MemorySource) -> list[MemoryChunk]:
        """Get all chunks for a file."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"""
                SELECT id, path, source, start_line, end_line, text, hash, embedding
                FROM {self.chunks_table_name} WHERE path = ? AND source = ?
                ORDER BY start_line
            """,
                (path, source.value),
            )

            chunks = []
            for row in cursor.fetchall():
                chunk_id, path_val, source_val, start, end, text, hash_val, emb_str = row
                # Parse embedding from JSON string
                embedding = None
                if emb_str:
                    try:
                        embedding = json.loads(emb_str)
                    except (json.JSONDecodeError, TypeError):
                        embedding = None

                chunks.append(
                    MemoryChunk(
                        id=chunk_id,
                        path=path_val,
                        source=MemorySource(source_val),
                        start_line=start,
                        end_line=end,
                        text=text,
                        hash=hash_val,
                        embedding=embedding,
                    ),
                )

            return chunks
        except Exception as e:
            logger.error(f"Failed to get file chunks for {path}: {e}")
            raise
        finally:
            cursor.close()

    async def vector_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector similarity search."""
        if not self.vector_enabled or not query:
            return []

        # Get query embedding
        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        cursor = self.conn.cursor()
        source_filter = ""
        params: list = []
        if sources:
            placeholders = ",".join("?" * len(sources))
            source_filter = f" AND c.source IN ({placeholders})"
            params = [s.value for s in sources]

        try:
            query_blob = self.vector_to_blob(query_embedding)

            # Correct SQLite-vec syntax for vector search with limit
            # vec0 requires 'k = ?' constraint for knn queries
            query_sql = f"""
                SELECT c.id, c.path, c.start_line, c.end_line, c.source, c.text, v.distance
                FROM {self.vector_table_name} v
                JOIN {self.chunks_table_name} c ON v.id = c.id
                WHERE v.embedding MATCH ?
                AND k = ?
            """
            query_params: list = [query_blob, limit]

            # Add source filter if specified
            if source_filter:
                query_sql += source_filter
                query_params.extend(params)

            # Order by distance (k constraint already limits results)
            query_sql += " ORDER BY v.distance"

            cursor.execute(query_sql, query_params)

            results = []
            for _, path, start, end, src, text, dist in cursor.fetchall():
                # Convert L2 distance to similarity score
                # For normalized vectors, L2 distance range is [0, 2]
                # Map to [1, 0] score range (higher score = more similar)
                score = max(0.0, 1.0 - dist / 2.0)
                snippet = text
                results.append(
                    MemorySearchResult(
                        path=path,
                        start_line=start,
                        end_line=end,
                        score=score,
                        snippet=snippet,
                        source=MemorySource(src),
                        raw_metric=dist,
                    ),
                )

            results.sort(key=lambda r: r.score, reverse=True)
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        finally:
            cursor.close()

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize query string for FTS5 search.

        Removes or escapes special characters that have special meaning in FTS5:
        - * (prefix match)
        - ? (not used in FTS5, but can cause issues)
        - " (phrase search, needs escaping)
        - : (column filter)
        - ^ (start of line anchor, not standard FTS5)
        - ' (single quote, causes syntax errors)
        - ` (backtick, can cause issues)
        - | (pipe, OR operator)
        - + (plus, can be used for required terms)
        - - (minus, NOT operator)
        - = (equals, can cause issues)
        - < > (angle brackets, comparison operators)
        - ! (exclamation, NOT operator variant)
        - @ # $ % & (other special chars)
        - "\"
        - / (slash, can interfere)
        - ; (semicolon, statement separator)
        - , (comma, can interfere with phrase parsing)

        Args:
            query: Raw query string

        Returns:
            Sanitized query string safe for FTS5
        """
        if not query:
            return ""

        # Remove FTS5 special characters that we don't want users to use
        # Keep only alphanumeric, spaces, periods, and underscores
        special_chars = [
            "*",
            "?",
            ":",
            "^",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "'",
            '"',
            "`",
            "|",
            "+",
            "-",
            "=",
            "<",
            ">",
            "!",
            "@",
            "#",
            "$",
            "%",
            "&",
            "\\",
            "/",
            ";",
            ",",
        ]
        cleaned = query
        for char in special_chars:
            cleaned = cleaned.replace(char, " ")

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned

    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform keyword search.

        Strategy:
        - FTS5 trigram (fast path): used when ALL terms >= 3 chars (trigram minimum).
        - LIKE (universal fallback): used when any term < 3 chars, covering CJK
          short words, single/double-char queries, and mixed-length queries.
        """
        if not self.fts_enabled:
            return []

        cleaned = self._sanitize_fts_query(query)
        if not cleaned:
            return []

        words = cleaned.split()
        if not words:
            return []

        # FTS5 trigram requires every term >= 3 characters
        if all(len(w) >= 3 for w in words):
            results = await self._fts_trigram_search(words, limit, sources)
            if results:
                return results

        # Universal fallback: LIKE-based substring search
        return await self._like_search(cleaned, words, limit, sources)

    async def _fts_trigram_search(
        self,
        words: list[str],
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """FTS5 trigram search. All terms must be >= 3 characters."""
        escaped_words = [w.replace('"', '""') for w in words]
        fts_query = " OR ".join(escaped_words)

        cursor = self.conn.cursor()
        source_filter = ""
        params: list = [fts_query]
        if sources:
            placeholders = ",".join("?" * len(sources))
            source_filter = f" AND fts.source IN ({placeholders})"
            params.extend([s.value for s in sources])
        params.append(limit)

        try:
            cursor.execute(
                f"""
                SELECT fts.id, fts.path, fts.start_line, fts.end_line,
                       fts.source, fts.text, rank
                FROM {self.fts_table_name} fts
                WHERE fts.text MATCH ?{source_filter}
                ORDER BY rank
                LIMIT ?
            """,
                params,
            )

            results = []
            for _, path, start, end, src, text, rank in cursor.fetchall():
                score = max(0.0, 1.0 / (1.0 + abs(rank)))
                results.append(
                    MemorySearchResult(
                        path=path,
                        start_line=start,
                        end_line=end,
                        score=score,
                        snippet=text,
                        source=MemorySource(src),
                        raw_metric=rank,
                    ),
                )
            results.sort(key=lambda r: r.score, reverse=True)
            return results
        except Exception as e:
            logger.error(f"FTS trigram search failed: {e}")
            return []
        finally:
            cursor.close()

    async def _like_search(
        self,
        phrase: str,
        words: list[str],
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """LIKE-based substring search with Python-side relevance scoring.

        Handles any term length and all languages (CJK, Latin, etc.).
        Scores results by: word-match ratio + full-phrase bonus.
        """
        cursor = self.conn.cursor()
        try:
            # Build OR conditions: match any individual word
            like_clauses = []
            params: list = []
            for word in words:
                like_clauses.append("c.text LIKE ?")
                params.append(f"%{word}%")

            where_clause = " OR ".join(like_clauses)

            source_filter = ""
            if sources:
                placeholders = ",".join("?" * len(sources))
                source_filter = f" AND c.source IN ({placeholders})"
                params.extend([s.value for s in sources])

            # Fetch extra candidates for re-ranking in Python
            fetch_limit = min(limit * 3, 200)
            params.append(fetch_limit)

            cursor.execute(
                f"""
                SELECT c.id, c.path, c.start_line, c.end_line, c.source, c.text
                FROM {self.chunks_table_name} c
                WHERE ({where_clause}){source_filter}
                LIMIT ?
            """,
                params,
            )

            results = []
            phrase_lower = phrase.lower()
            words_lower = [w.lower() for w in words]
            n_words = len(words)

            for _, path, start, end, src, text in cursor.fetchall():
                text_lower = text.lower()

                # Base score: proportion of query words found in text
                match_count = sum(1 for w in words_lower if w in text_lower)
                base_score = match_count / n_words

                # Bonus: full phrase appears as contiguous substring
                phrase_bonus = 0.2 if n_words > 1 and phrase_lower in text_lower else 0.0

                score = min(1.0, base_score * 0.8 + phrase_bonus)

                results.append(
                    MemorySearchResult(
                        path=path,
                        start_line=start,
                        end_line=end,
                        score=score,
                        snippet=text,
                        source=MemorySource(src),
                    ),
                )

            # Sort by score descending, return top `limit`
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"LIKE search failed: {e}")
            return []
        finally:
            cursor.close()

    async def hybrid_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
    ) -> list[MemorySearchResult]:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query text
            limit: Maximum number of results
            sources: Optional list of sources to filter
            vector_weight: Weight for vector search results (0.0-1.0).
                          Keyword weight = 1.0 - vector_weight.
            candidate_multiplier: Multiplier for candidate pool size.

        Returns:
            List of search results sorted by combined relevance score
        """
        assert 0.0 <= vector_weight <= 1.0, f"vector_weight must be between 0 and 1, got {vector_weight}"

        candidates = min(200, max(1, int(limit * candidate_multiplier)))
        text_weight = 1.0 - vector_weight

        # Perform search based on enabled backends
        if self.vector_enabled and self.fts_enabled:
            keyword_results = await self.keyword_search(query, candidates, sources)
            vector_results = await self.vector_search(query, candidates, sources)

            # Log original vector results
            logger.info("\n=== Vector Search Results ===")
            for i, r in enumerate(vector_results[:10], 1):
                snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                logger.info(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

            # Log original keyword results
            logger.info("\n=== Keyword Search Results ===")
            for i, r in enumerate(keyword_results[:10], 1):
                snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                logger.info(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

            if not keyword_results:
                return vector_results[:limit]
            elif not vector_results:
                return keyword_results[:limit]
            else:
                merged = self._merge_hybrid_results(
                    vector=vector_results,
                    keyword=keyword_results,
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                )

                # Log merged results
                logger.info("\n=== Merged Hybrid Results ===")
                for i, r in enumerate(merged[:10], 1):
                    snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                    logger.info(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

                return merged[:limit]
        elif self.vector_enabled:
            vector_results = await self.vector_search(query, limit, sources)
            return vector_results
        elif self.fts_enabled:
            keyword_results = await self.keyword_search(query, limit, sources)
            return keyword_results
        else:
            return []

    @staticmethod
    def _merge_hybrid_results(
        vector: list[MemorySearchResult],
        keyword: list[MemorySearchResult],
        vector_weight: float,
        text_weight: float,
    ) -> list[MemorySearchResult]:
        """Merge vector and keyword search results with weighted scoring."""
        merged: dict[str, MemorySearchResult] = {}

        # Process vector results
        for result in vector:
            result.score = result.score * vector_weight
            merged[result.merge_key] = result

        # Process keyword results
        for result in keyword:
            key = result.merge_key
            if key in merged:
                merged[key].score += result.score * text_weight
            else:
                result.score = result.score * text_weight
                merged[key] = result

        # Sort by score and return
        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def clear_all(self):
        """Clear all indexed data."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            cursor.execute(f"DELETE FROM {self.files_table_name}")
            cursor.execute(f"DELETE FROM {self.chunks_table_name}")

            if self.vector_enabled:
                cursor.execute(f"DELETE FROM {self.vector_table_name}")

            if self.fts_enabled:
                cursor.execute(f"DELETE FROM {self.fts_table_name}")

            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Failed to clear all data: {e}")
            raise
        finally:
            cursor.close()

    async def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
        await super().close()
