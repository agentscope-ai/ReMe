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
        chunk_chars: int = 2000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.chunk_chars = max(256, chunk_chars)
        self.min_element_size = max(64, int(self.chunk_chars * 0.05))

    class Node:
        """AST node for JSON tree construction."""

        def __init__(self) -> None:
            self.type = None  # string, number, boolean, array, object, null
            self.start_line = None
            self.end_line = None
            self.values = None  # type-dependent:
            #   string: indent=None JSON text of the primitive / small container
            #   array: list[Node]
            #   object: list[tuple[str, Node]] (key, child) in text order
            #   null: None

    _decoder = json.JSONDecoder()

    #: Number of leaves between size-calibration checkpoints.
    _CALIBRATE_EVERY = 64

    class _SizeNode:
        """Mutable pruned-tree node with cached compact-JSON size.

        Tracks the size of ``json.dumps(data, ensure_ascii=False)`` for
        a pruned subtree incrementally, avoiding repeated full
        serialisation.  Size formula:

        - Object ``{"k": v}`` → ``2 + Σ(len(dumps(k)) + 2 + child) + 2*(n-1)``
        - Array ``[v1, v2]``    → ``2 + Σ(child) + 2*(n-1)``
        """

        __slots__ = ("is_object", "items", "size")

        def __init__(self, is_object: bool) -> None:
            self.is_object = is_object
            self.items: list = []  # (step, _SizeNode | int)
            self.size = 2  # {} or []

        def add_leaf(self, path: list, leaf_size: int) -> None:
            """Add a leaf at *path* with pre-computed serialised *leaf_size*."""
            if not path:
                return

            step = path[0]
            rest = path[1:]

            # DFS order: if the last child shares this step, merge into it.
            if self.items and self.items[-1][0] == step:
                child = self.items[-1][1]
                if isinstance(child, JsonFileChunker._SizeNode):  # pylint: disable=protected-access
                    old = child.size
                    child.add_leaf(rest, leaf_size)
                    self.size += child.size - old
                return

            # New child
            if rest:
                child = JsonFileChunker._SizeNode(isinstance(rest[0], str))  # pylint: disable=protected-access
                child.add_leaf(rest, leaf_size)
                child_size = child.size
            else:
                child = leaf_size
                child_size = leaf_size

            if self.is_object:
                # "key": value  →  len(dumps(key)) + 2 + child_size
                item_size = len(json.dumps(step, ensure_ascii=False)) + 2 + child_size
            else:
                item_size = child_size

            sep = 2 if self.items else 0  # ", " between siblings
            self.items.append((step, child))
            self.size += item_size + sep

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
            if len(raw_text) <= self.chunk_chars:
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
            else:
                lines = raw_text.split("\n", keepends=True)
                chunked_lines = [{"text": "", "start_line": 1, "end_line": 1}]
                for idx, line in enumerate(lines):
                    if len(chunked_lines[-1]["text"]) + len(line) > self.chunk_chars:
                        chunked_lines.append({"text": "", "start_line": idx + 1, "end_line": idx + 1})
                    chunked_lines[-1]["text"] += line
                    chunked_lines[-1]["end_line"] = idx + 1
                chunks = [
                    FileChunk(
                        path=rel_path,
                        start_line=x["start_line"],
                        end_line=x["end_line"],
                        text=x["text"],
                    ).set_hash_id()
                    for x in chunked_lines
                ]
                return (
                    FileNode(path=rel_path, st_mtime=stat.st_mtime, chunk_ids=[c.id for c in chunks]),
                    chunks,
                )

        # Build a structure-aware tree whose node line ranges align with
        # *raw_text*.  Leaves are atomic fragments below ``min_element_size``.
        root, _, _ = self._build_tree(raw_text, 0, 1, data)

        # Convert the tree into self-contained JSON chunks.
        chunk_pieces = self._node_to_chunks(root)
        if not chunk_pieces:
            return FileNode(path=rel_path, st_mtime=stat.st_mtime), []

        file_chunks: list[FileChunk] = []
        for cdata, start_line, end_line in chunk_pieces:
            text = json.dumps(cdata, ensure_ascii=False, indent=None)
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
        )
        return node, file_chunks

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    @staticmethod
    def _skip_ws(text: str, idx: int, line: int) -> tuple[int, int]:
        """Advance *idx* past whitespace, counting newlines."""
        while idx < len(text) and text[idx] in " \t\n\r":
            if text[idx] == "\n":
                line += 1
            idx += 1
        return idx, line

    def _build_tree(
        self,
        text: str,
        idx: int,
        line: int,
        data: Any,
    ) -> tuple[Node, int, int]:
        """Recursively build a ``Node`` tree from *data* aligned to *text*.

        Returns ``(node, next_idx, next_line)``.
        """
        idx, line = self._skip_ws(text, idx, line)
        start_line = line

        # --- primitive: serialise and store as leaf ---
        if not isinstance(data, (dict, list)):
            serialized = json.dumps(data, ensure_ascii=False)
            _, end_idx = self._decoder.raw_decode(text, idx)
            end_line = line + text[idx:end_idx].count("\n")
            node = self.Node()
            node.type = "string"
            node.start_line = start_line
            node.end_line = end_line
            node.values = serialized
            return node, end_idx, end_line

        # --- empty container: always a leaf (2 bytes) ---
        if not data:
            serialized = "{}" if isinstance(data, dict) else "[]"
            _, end_idx = self._decoder.raw_decode(text, idx)
            end_line = line + text[idx:end_idx].count("\n")
            node = self.Node()
            node.type = "string"
            node.start_line = start_line
            node.end_line = end_line
            node.values = serialized
            return node, end_idx, end_line

        # ---- non-empty container: recurse into children ----
        if isinstance(data, dict):
            idx += 1  # skip '{'
            children = []  # list[tuple[str, Node]]
            while True:
                idx, line = self._skip_ws(text, idx, line)
                if idx >= len(text) or text[idx] == "}":
                    if idx < len(text):
                        idx += 1  # skip '}'
                    break
                # Parse key directly from text to guarantee text-order alignment
                key, idx = self._decoder.raw_decode(text, idx)
                idx, line = self._skip_ws(text, idx, line)
                idx += 1  # skip ':'
                child, idx, line = self._build_tree(text, idx, line, data[key])
                children.append((key, child))
                idx, line = self._skip_ws(text, idx, line)
                if idx < len(text) and text[idx] == ",":
                    idx += 1
            end_line = line

            # Promote to leaf if compact serialisation is small enough.
            # Only pay one json.dumps for containers that *become* leaves.
            reconstructed = self._reconstruct_object(children)
            serialized = json.dumps(reconstructed, ensure_ascii=False)
            if len(serialized) <= self.min_element_size:
                node = self.Node()
                node.type = "string"
                node.start_line = start_line
                node.end_line = end_line
                node.values = serialized
                return node, idx, end_line

            node = self.Node()
            node.type = "object"
            node.start_line = start_line
            node.end_line = end_line
            node.values = children
            return node, idx, end_line

        else:  # list
            idx += 1  # skip '['
            children_list = []  # list[Node]
            for i, val in enumerate(data):
                idx, line = self._skip_ws(text, idx, line)
                child, idx, line = self._build_tree(text, idx, line, val)
                children_list.append(child)
                idx, line = self._skip_ws(text, idx, line)
                if i < len(data) - 1 and idx < len(text) and text[idx] == ",":
                    idx += 1
            idx, line = self._skip_ws(text, idx, line)
            if idx < len(text) and text[idx] == "]":
                idx += 1
            end_line = line

            reconstructed = self._reconstruct_array(children_list)
            serialized = json.dumps(reconstructed, ensure_ascii=False)
            if len(serialized) <= self.min_element_size:
                node = self.Node()
                node.type = "string"
                node.start_line = start_line
                node.end_line = end_line
                node.values = serialized
                return node, idx, end_line

            node = self.Node()
            node.type = "array"
            node.start_line = start_line
            node.end_line = end_line
            node.values = children_list
            return node, idx, end_line

    # ------------------------------------------------------------------
    # Tree -> chunks
    # ------------------------------------------------------------------

    def _reconstruct_data(self, node: Node) -> Any:
        """Reconstruct the original Python data from a tree node."""
        if node.type == "string":
            return json.loads(node.values)
        if node.type == "object":
            return self._reconstruct_object(node.values)
        return self._reconstruct_array(node.values)

    def _reconstruct_object(self, children):
        return {k: self._reconstruct_data(v) for k, v in children}

    def _reconstruct_array(self, children):
        return [self._reconstruct_data(v) for v in children]

    def _node_to_chunks(self, node: Node) -> list[tuple[Any, int, int]]:
        """Split a tree into path-wrapped chunks via DFS greedy grouping.

        Performs depth-first search over the tree's leaves.  Accumulates
        leaves until the pruned subtree (which preserves the root-to-leaf
        path for every accumulated leaf) exceeds ``chunk_chars``.  Each
        chunk is a valid JSON value whose nesting mirrors the original
        structure.

        Size is tracked incrementally via :class:`_SizeNode` (O(depth)
        per leaf) instead of re-serialising the entire pruned tree on
        every addition.  A calibration checkpoint runs every
        ``_CALIBRATE_EVERY`` leaves to guard against drift.
        """
        leaves = list(self._collect_leaves(node))
        if not leaves:
            return []

        # Root-level leaf — single chunk, no _SizeNode needed.
        if node.type == "string":
            data = json.loads(node.values)
            return [(data, node.start_line, node.end_line)]

        root_is_object = node.type == "object"
        chunks: list[tuple[Any, int, int]] = []
        current: list[tuple[Any, int, int, list, int]] = []
        tree = self._SizeNode(root_is_object)

        for leaf in leaves:
            current.append(leaf)
            tree.add_leaf(leaf[3], leaf[4])  # path, leaf_size

            # Periodic calibration.
            if len(current) % self._CALIBRATE_EVERY == 0:
                pruned = self._build_pruned_tree(current)
                tree.size = len(json.dumps(pruned, ensure_ascii=False))

            if tree.size > self.chunk_chars and len(current) > 1:
                # Adding this leaf exceeds the limit; emit without it.
                current.pop()
                pruned = self._build_pruned_tree(current)
                chunks.append((pruned, current[0][1], current[-1][2]))
                current = [leaf]
                tree = self._SizeNode(root_is_object)
                tree.add_leaf(leaf[3], leaf[4])

        if current:
            pruned = self._build_pruned_tree(current)
            chunks.append((pruned, current[0][1], current[-1][2]))

        return chunks

    def _collect_leaves(
        self,
        node: Node,
        path: list | None = None,
    ) -> Any:
        """DFS generator yielding ``(data, start_line, end_line, path, leaf_size)``.

        ``path`` is a list of steps from root to leaf: strings for object
        keys, integers for array element indices.  ``leaf_size`` is the
        compact serialised length of the leaf value (pre-computed from
        ``node.values`` to avoid redundant ``json.dumps`` calls).
        """
        if path is None:
            path = []
        if node.type == "string":
            yield (
                json.loads(node.values),
                node.start_line,
                node.end_line,
                list(path),
                len(node.values),
            )
            return
        if node.type == "object":
            for key, child in node.values:
                path.append(key)
                yield from self._collect_leaves(child, path)
                path.pop()
        else:  # array
            for idx, child in enumerate(node.values):
                path.append(idx)
                yield from self._collect_leaves(child, path)
                path.pop()

    def _build_pruned_tree(
        self,
        leaves: list[tuple[Any, int, int, list, int]],
    ) -> Any:
        """Build the minimal pruned subtree containing all *leaves*.

        For each leaf the path from root is preserved as wrapping:
        object ancestors keep only the relevant key, array ancestors
        keep only the relevant elements in original order.
        """
        if not leaves:
            return None

        # Root-level leaf (path is empty) — return value directly.
        if not leaves[0][3]:
            return leaves[0][0]

        first_step = leaves[0][3][0]
        if isinstance(first_step, str):
            # Object root: group by key, preserving first-seen order.
            groups: dict[str, list] = {}
            order: list[str] = []
            for data, sl, el, path, _sz in leaves:
                key = path[0]
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append((data, sl, el, path[1:], _sz))
            return {k: self._build_pruned_tree(groups[k]) for k in order}
        else:
            # Array root: group by original index, preserving order.
            groups_arr: dict[int, list] = {}
            order_arr: list[int] = []
            for data, sl, el, path, _sz in leaves:
                idx = path[0]
                if idx not in groups_arr:
                    groups_arr[idx] = []
                    order_arr.append(idx)
                groups_arr[idx].append((data, sl, el, path[1:], _sz))
            return [self._build_pruned_tree(groups_arr[i]) for i in order_arr]
