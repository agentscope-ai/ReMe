"""JSON file chunker — structure-aware chunking preserving key paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base_file_chunker import BaseFileChunker
from ..component_registry import R
from ...schema import FileChunk, FileNode


@R.register("json")
class JsonFileChunker(BaseFileChunker):
    """Chunker for structured JSON files.

    Splits JSON into smaller sub-dicts while preserving nested key paths.
    Each chunk is a valid JSON object whose keys mirror the original
    structure.  Lists can optionally be converted to index-keyed dicts
    for better splitting granularity.

    Size is measured in serialized character count (``len(json.dumps(...))``).
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        max_chunk_size: int = 2000,
        min_chunk_size: int | None = None,
        convert_lists: bool = True,
        ensure_ascii: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.max_chunk_size = max(64, max_chunk_size)
        self.min_chunk_size = min_chunk_size if min_chunk_size is not None else max(self.max_chunk_size - 200, 50)
        self.convert_lists = convert_lists
        self.ensure_ascii = ensure_ascii

    # ------------------------------------------------------------------
    # Size / dict helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_size(data: Any) -> int:
        """Character count of the serialised JSON object."""
        return len(json.dumps(data))

    @staticmethod
    def _set_nested_dict(d: dict[str, Any], path: list[str], value: Any) -> None:
        """Set a value in a nested dict following *path*, creating intermediates."""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        """Recursively convert lists to index-keyed dicts."""
        if isinstance(data, dict):
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        if isinstance(data, list):
            return {str(i): self._list_to_dict_preprocessing(item) for i, item in enumerate(data)}
        return data

    # ------------------------------------------------------------------
    # Core splitting
    # ------------------------------------------------------------------

    def _json_split(
        self,
        data: Any,
        current_path: list[str] | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Recursively split *data* into sub-dicts bounded by max_chunk_size."""
        current_path = current_path or []
        chunks = chunks if chunks is not None else [{}]

        if isinstance(data, dict) and data:
            for key, value in data.items():
                new_path = [*current_path, key]
                chunk_size = self._json_size(chunks[-1])
                size = self._json_size({key: value})
                remaining = self.max_chunk_size - chunk_size

                if size < remaining:
                    self._set_nested_dict(chunks[-1], new_path, value)
                else:
                    if chunk_size >= self.min_chunk_size:
                        chunks.append({})
                    self._json_split(value, new_path, chunks)

        elif current_path:
            self._set_nested_dict(chunks[-1], current_path, data)

        return chunks

    def _split_json_data(self, data: Any) -> list[dict[str, Any]]:
        """Preprocess (optional) and split *data*, removing trailing empty chunk."""
        processed = self._list_to_dict_preprocessing(data) if self.convert_lists else data
        chunks = self._json_split(processed)
        if chunks and not chunks[-1]:
            chunks.pop()
        return chunks

    # ------------------------------------------------------------------
    # Line-range mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _build_line_index(text: str, data: Any) -> dict[str, list[int]]:
        """Walk *data* and map each terminal key to its 1-based line numbers.

        Returns ``{terminal_key: [line, ...]}`` so duplicate keys across
        different branches are all recorded.
        """
        lines = text.split("\n")
        index: dict[str, list[int]] = {}

        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    for i, line in enumerate(lines, 1):
                        if f'"{k}"' in line:
                            index.setdefault(k, [])
                            if i not in index[k]:
                                index[k].append(i)
                            break
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        _walk(data)
        return index

    def _compute_line_range(
        self,
        chunk: dict[str, Any],
        line_index: dict[str, list[int]],
        used: dict[str, int],
        total_lines: int,
    ) -> tuple[int, int]:
        """Compute (start_line, end_line) for *chunk* using terminal-key lookup."""
        chunk_lines: list[int] = []

        def _collect(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in line_index:
                        pos = used.get(k, 0)
                        lines_list = line_index[k]
                        if pos < len(lines_list):
                            chunk_lines.append(lines_list[pos])
                            used[k] = pos + 1
                    _collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    _collect(item)

        _collect(chunk)
        if chunk_lines:
            return min(chunk_lines), max(chunk_lines)
        return 1, total_lines or 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chunk(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Read and chunk a JSON file at *path*."""
        file_path = Path(path)
        stat = file_path.stat()
        rel_path = self.to_workspace_relative(path)

        raw_text = file_path.read_text(encoding=self.encoding)
        if not raw_text.strip():
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Malformed JSON: fall back to treating the whole file as one chunk.
            total_lines = raw_text.count("\n") + 1
            chunk = FileChunk(
                path=rel_path,
                start_line=1,
                end_line=total_lines,
                text=raw_text,
            ).set_hash_id()
            return (
                FileNode(path=rel_path, st_mtime=stat.st_mtime, chunk_ids=[chunk.id]),
                [chunk],
            )

        json_chunks = self._split_json_data(data)
        if not json_chunks:
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        # Build line index for mapping chunks back to source positions.
        line_index = self._build_line_index(raw_text, data)
        used: dict[str, int] = {}
        total_lines = raw_text.count("\n") + 1

        file_chunks: list[FileChunk] = []
        for jc in json_chunks:
            text = json.dumps(jc, ensure_ascii=self.ensure_ascii, indent=2)
            start_line, end_line = self._compute_line_range(jc, line_index, used, total_lines)
            file_chunks.append(
                FileChunk(
                    path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                ).set_hash_id(),
            )

        node = FileNode(
            path=rel_path,
            st_mtime=stat.st_mtime,
            chunk_ids=[c.id for c in file_chunks],
            links=[],
        )
        return node, file_chunks
