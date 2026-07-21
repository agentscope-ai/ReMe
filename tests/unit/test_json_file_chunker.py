"""Tests for JsonFileChunker — DFS greedy pruned-tree chunking."""

# pylint: disable=protected-access,missing-class-docstring
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-variable

import asyncio
import json
import os
import tempfile

import pytest

from reme.components.file_chunker import JsonFileChunker

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _deep_merge(target: dict, source: dict) -> dict:
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v
    return target


def _merge_chunks(chunks) -> dict:
    """Deep-merge all chunk texts into a single dict."""
    merged: dict = {}
    for c in chunks:
        _deep_merge(merged, json.loads(c.text))
    return merged


def _deep_merge_full(a, b):
    """Recursively merge *b* into *a*, handling both dicts and arrays."""
    if isinstance(a, dict) and isinstance(b, dict):
        result = dict(a)
        for k, v in b.items():
            if k in result:
                result[k] = _deep_merge_full(result[k], v)
            else:
                result[k] = v
        return result
    if isinstance(a, list) and isinstance(b, list):
        result = list(a)
        for i, v in enumerate(b):
            if i < len(result):
                result[i] = _deep_merge_full(result[i], v)
            else:
                result.append(v)
        return result
    return b


def _all_leaf_values(data):
    """Yield all scalar leaf values from a JSON structure in DFS order."""
    if isinstance(data, dict):
        for v in data.values():
            yield from _all_leaf_values(v)
    elif isinstance(data, list):
        for v in data:
            yield from _all_leaf_values(v)
    else:
        yield data


def _is_single_leaf(val) -> bool:
    """True if *val* is a chain of single-element containers ending in a scalar."""
    while isinstance(val, (dict, list)):
        if len(val) != 1:
            return False
        val = next(iter(val.values())) if isinstance(val, dict) else val[0]
    return True


@pytest.fixture
def make_json():
    """Factory that writes temp JSON files and cleans up afterwards."""
    paths: list[str] = []

    def _make(data=None, raw: str | None = None) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if raw is not None:
                f.write(raw)
            elif data is not None:
                json.dump(data, f, indent=2)
        paths.append(path)
        return path

    yield _make
    for p in paths:
        os.unlink(p)


def _build_tree(chunker, data):
    """Build a tree from *data* and return the root node."""
    text = json.dumps(data, indent=2)
    root, _, _ = chunker._build_tree(text, 0, 1, data)
    return root


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------


class TestBasicChunking:
    def test_empty_file(self, make_json):
        path = make_json(raw="")
        _, chunks = _run(JsonFileChunker().chunk(path))
        assert chunks == []

    @pytest.mark.parametrize("data", [{}, []])
    def test_empty_container(self, make_json, data):
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker().chunk(path))
        assert len(chunks) == 1
        assert json.loads(chunks[0].text) == data

    def test_small_json_one_chunk(self, make_json):
        data = {"name": "Alice", "age": 30}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=5000).chunk(path))
        assert len(chunks) == 1
        assert json.loads(chunks[0].text) == data

    def test_large_json_multiple_chunks(self, make_json):
        data = {f"key_{i}": f"value_{i}" * 20 for i in range(50)}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=500).chunk(path))
        assert len(chunks) > 1
        all_keys = {k for c in chunks for k in json.loads(c.text)}
        assert all_keys == set(data)

    def test_nested_deep_merge(self, make_json):
        data = {
            "section_a": {"name": "Alice", "score": 95, "detail": "x" * 200},
            "section_b": {"name": "Bob", "score": 88, "detail": "y" * 200},
            "section_c": {"name": "Charlie", "score": 72, "detail": "z" * 200},
        }
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=400).chunk(path))
        assert len(chunks) > 1
        assert _merge_chunks(chunks) == data


# ---------------------------------------------------------------------------
# Top-level types
# ---------------------------------------------------------------------------


class TestTopLevelTypes:
    @pytest.mark.parametrize(
        "raw, expected",
        [("42", 42), ('"hello"', "hello"), ("null", None), ("true", True)],
    )
    def test_primitives(self, make_json, raw, expected):
        path = make_json(raw=raw)
        _, chunks = _run(JsonFileChunker().chunk(path))
        assert len(chunks) == 1
        assert json.loads(chunks[0].text) == expected

    def test_top_level_array(self, make_json):
        path = make_json(data=[{"a": 1}, {"b": 2}])
        _, chunks = _run(JsonFileChunker(chunk_chars=5000).chunk(path))
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(json.loads(c.text), list)


# ---------------------------------------------------------------------------
# Line-range mapping
# ---------------------------------------------------------------------------


class TestLineRanges:
    def test_monotonic(self, make_json):
        data = {f"key_{i}": f"val_{i}" for i in range(30)}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=300).chunk(path))
        assert len(chunks) > 1
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line
        assert chunks[0].start_line <= 3

    def test_single_chunk_covers_file(self, make_json):
        data = {"alpha": {"x": 1}, "beta": {"y": 2}, "gamma": {"z": 3}}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=5000).chunk(path))
        assert len(chunks) == 1
        assert chunks[0].start_line >= 1
        assert chunks[0].end_line >= 5

    def test_compact_format(self, make_json):
        raw = (
            "{\n"
            '  "alice": {"name": "Alice", "bio": "' + "x" * 200 + '"},\n'
            '  "bob": {"name": "Bob", "bio": "' + "y" * 200 + '"}\n'
            "}"
        )
        path = make_json(raw=raw)
        _, chunks = _run(JsonFileChunker(chunk_chars=300).chunk(path))
        assert len(chunks) >= 2
        assert _merge_chunks(chunks) == {
            "alice": {"name": "Alice", "bio": "x" * 200},
            "bob": {"name": "Bob", "bio": "y" * 200},
        }
        assert chunks[0].start_line == 2
        assert chunks[-1].end_line == 3
        for i in range(1, len(chunks)):
            assert chunks[i].start_line >= chunks[i - 1].start_line

    def test_minified_all_line_1(self, make_json):
        path = make_json(raw='{"a": 1, "b": 2, "c": 3}')
        _, chunks = _run(JsonFileChunker(chunk_chars=50).chunk(path))
        for c in chunks:
            assert c.start_line == 1 and c.end_line == 1

    def test_branch_order(self, make_json):
        """User chunk starts before assistant chunk (DFS order)."""
        data = {
            "user": {"name": "Alice", "bio": "x" * 200},
            "assistant": {"name": "Bob", "bio": "y" * 200},
        }
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=150).chunk(path))
        assert len(chunks) >= 2
        # Find first chunk that has only user (not assistant)
        user_only = [
            c.start_line for c in chunks if "user" in json.loads(c.text) and "assistant" not in json.loads(c.text)
        ]
        asst_chunks = [c.start_line for c in chunks if "assistant" in json.loads(c.text)]
        assert user_only and asst_chunks
        assert min(user_only) < min(asst_chunks)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_malformed_fallback(self, make_json):
        path = make_json(raw='{"bad": json}')
        _, chunks = _run(JsonFileChunker().chunk(path))
        assert len(chunks) == 1
        assert chunks[0].start_line == 1

    def test_chunk_chars_floor(self):
        assert JsonFileChunker(chunk_chars=10).chunk_chars == 256

    def test_min_element_size_formula(self):
        assert JsonFileChunker(chunk_chars=2000).min_element_size == 100  # max(64, 2000*0.05)
        assert JsonFileChunker(chunk_chars=100).min_element_size == 64  # max(64, 100*0.05)
        assert JsonFileChunker(chunk_chars=10000).min_element_size == 500  # max(64, 10000*0.05)

    def test_deeply_nested(self, make_json):
        data = {"level1": {"level2": {"level3": {"value": "deep" * 100}}}}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=200).chunk(path))
        assert len(chunks) >= 1
        merged = _merge_chunks(chunks)
        assert merged["level1"]["level2"]["level3"]["value"] == "deep" * 100

    def test_unicode(self, make_json):
        data = {"greeting": "你好世界", "emoji": "🎉🎊", "mixed": "hello世界"}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker().chunk(path))
        assert json.loads(chunks[0].text) == data

    def test_chunk_size_respected(self, make_json):
        data = {f"k{i}": "v" * 50 for i in range(40)}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=500).chunk(path))
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 500 * 3

    def test_all_valid_json(self, make_json):
        data = {
            "users": [{"name": f"u{i}", "desc": "x" * 100, "tags": [f"t{j}" for j in range(5)]} for i in range(10)],
            "metadata": {"count": 10, "version": "1.0"},
        }
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=300).chunk(path))
        assert len(chunks) > 1
        for c in chunks:
            json.loads(c.text)  # raises if invalid

    def test_cjk_single_chunk(self, make_json):
        data = {f"键名_{i}": f"中文值内容_{i}" for i in range(8)}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=400).chunk(path))
        assert len(chunks) == 1

    def test_same_key_different_levels(self, make_json):
        data = {"a": {"a": 1, "b": 2}, "b": {"a": 3}}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=200).chunk(path))
        assert _merge_chunks(chunks) == data

    def test_string_value_looking_like_key(self, make_json):
        data = {"first": "name", "name": "Alice"}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=100).chunk(path))
        assert _merge_chunks(chunks) == data

    def test_oversized_leaf_atomic(self, make_json):
        """A single leaf exceeding chunk_chars still gets emitted."""
        data = {"big": "x" * 5000}
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=500).chunk(path))
        assert len(chunks) == 1  # "big" is a leaf (primitive), can't split


# ---------------------------------------------------------------------------
# Array chunking
# ---------------------------------------------------------------------------


class TestArrayChunking:
    def test_large_array_all_ids_present(self, make_json):
        data = [{"id": i, "data": "x" * 200} for i in range(20)]
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=300).chunk(path))
        assert len(chunks) > 1
        all_ids = {elem["id"] for c in chunks for elem in json.loads(c.text) if isinstance(elem, dict) and "id" in elem}
        assert all_ids == set(range(20))

    def test_nested_array_chunking(self, make_json):
        """Array of objects with nested structure is chunked correctly."""
        data = [{"name": f"item_{i}", "values": list(range(10))} for i in range(30)]
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=400).chunk(path))
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(json.loads(c.text), list)


# ---------------------------------------------------------------------------
# Internal: _build_tree, _reconstruct_data
# ---------------------------------------------------------------------------


class TestInternals:
    def test_build_tree_structure(self):
        # Use large strings so containers exceed min_element_size (250 for chunk_chars=5000)
        data = {"aaa": "x" * 500, "bbb": {"ccc": "y" * 500}}
        chunker = JsonFileChunker(chunk_chars=5000)
        root = _build_tree(chunker, data)
        assert root.type == "object"
        assert [k for k, _ in root.values] == ["aaa", "bbb"]
        # "aaa" is a leaf on line 2
        assert root.values[0][1].type == "string"
        assert root.values[0][1].start_line == 2
        # "bbb" is an internal object on lines 3-5
        node_b = root.values[1][1]
        assert node_b.type == "object"
        assert node_b.start_line == 3
        assert [k for k, _ in node_b.values] == ["ccc"]

    def test_reconstruct_data(self):
        data = {"a": 1, "b": [1, 2, {"c": 3}], "d": {"e": "hello"}}
        chunker = JsonFileChunker(chunk_chars=5000)
        root = _build_tree(chunker, data)
        assert chunker._reconstruct_data(root) == data

    def test_node_to_chunks_leaf(self):
        chunker = JsonFileChunker()
        node = chunker.Node()
        node.type = "string"
        node.start_line = 1
        node.end_line = 1
        node.values = json.dumps({"a": 1}, ensure_ascii=False)
        chunks = chunker._node_to_chunks(node)
        assert len(chunks) == 1
        data, s, e = chunks[0]
        assert data == {"a": 1}
        assert s == 1 and e == 1


# ---------------------------------------------------------------------------
# _SizeNode — incremental size tracking
# ---------------------------------------------------------------------------


class TestSizeNode:
    def test_object_size_exact(self):
        """_SizeNode matches json.dumps for flat objects."""
        chunker = JsonFileChunker()
        tree = chunker._SizeNode(True)
        tree.add_leaf(["a"], len(json.dumps(1, ensure_ascii=False)))
        assert tree.size == len(json.dumps({"a": 1}, ensure_ascii=False))
        tree.add_leaf(["b"], len(json.dumps(2, ensure_ascii=False)))
        assert tree.size == len(json.dumps({"a": 1, "b": 2}, ensure_ascii=False))

    def test_array_size_exact(self):
        chunker = JsonFileChunker()
        tree = chunker._SizeNode(False)
        tree.add_leaf([0], len(json.dumps(1, ensure_ascii=False)))
        tree.add_leaf([1], len(json.dumps("x", ensure_ascii=False)))
        assert tree.size == len(json.dumps([1, "x"], ensure_ascii=False))

    def test_nested_merge_into_existing_child(self):
        """Two leaves sharing an object key merge correctly."""
        chunker = JsonFileChunker()
        tree = chunker._SizeNode(True)
        sz1 = len(json.dumps(1, ensure_ascii=False))
        sz2 = len(json.dumps(2, ensure_ascii=False))
        tree.add_leaf(["a", "x"], sz1)
        tree.add_leaf(["a", "y"], sz2)
        assert tree.size == len(
            json.dumps({"a": {"x": 1, "y": 2}}, ensure_ascii=False),
        )

    def test_accuracy_against_real_chunker(self):
        """Verify _SizeNode matches json.dumps across many accumulation points."""
        data = {f"key_{i}": f"val_{i}_" + "x" * 100 for i in range(100)}
        chunker = JsonFileChunker(chunk_chars=500)
        root = _build_tree(chunker, data)
        leaves = list(chunker._collect_leaves(root))
        tree = chunker._SizeNode(True)
        for i, leaf in enumerate(leaves):
            tree.add_leaf(leaf[3], leaf[4])
            if (i + 1) % 10 == 0 or i == len(leaves) - 1:
                pruned = chunker._build_pruned_tree(leaves[: i + 1])
                actual = len(json.dumps(pruned, ensure_ascii=False))
                assert tree.size == actual, f"drift at leaf {i}: {tree.size} vs {actual}"

    def test_nested_accuracy(self):
        """Accuracy with nested arrays + objects."""
        data = {
            "a": [{"x": i, "y": i * 2, "z": "hello"} for i in range(50)],
            "b": {"c": 1, "d": "test" * 50},
        }
        chunker = JsonFileChunker(chunk_chars=400)
        root = _build_tree(chunker, data)
        leaves = list(chunker._collect_leaves(root))
        tree = chunker._SizeNode(True)
        for i, leaf in enumerate(leaves):
            tree.add_leaf(leaf[3], leaf[4])
            if (i + 1) % 15 == 0 or i == len(leaves) - 1:
                pruned = chunker._build_pruned_tree(leaves[: i + 1])
                actual = len(json.dumps(pruned, ensure_ascii=False))
                assert tree.size == actual, f"drift at leaf {i}: {tree.size} vs {actual}"


# ---------------------------------------------------------------------------
# DFS greedy algorithm — path wrapping & order
# ---------------------------------------------------------------------------


class TestDfsAlgorithm:
    def test_path_wrapping_preserved(self):
        """Each chunk preserves the root-to-leaf nesting structure."""
        data = {"a": ["b" * 200, {"c": 1, "d": 2}, "c" * 200, 1], "b": 10}
        chunker = JsonFileChunker(chunk_chars=300)
        chunker.min_element_size = 1  # force {"c":1,"d":2} to be internal
        root = _build_tree(chunker, data)
        chunks = chunker._node_to_chunks(root)
        assert len(chunks) >= 2
        for cdata, sl, el in chunks:
            # Every chunk must be a dict with key "a" (path preserved)
            assert isinstance(cdata, dict)
            assert "a" in cdata
            assert isinstance(cdata["a"], list)

    def test_calibration_during_chunking(self):
        """Calibration checkpoint keeps size accurate for large inputs."""
        data = {f"k{i}": "v" * 50 for i in range(200)}
        chunker = JsonFileChunker(chunk_chars=500)
        root = _build_tree(chunker, data)
        chunks = chunker._node_to_chunks(root)
        assert len(chunks) > 1
        # Every chunk (except possibly the last single-leaf) should be ≤ chunk_chars
        for cdata, _, _ in chunks:
            size = len(json.dumps(cdata, ensure_ascii=False))
            # Allow single oversized leaf
            if size > chunker.chunk_chars:
                # This must be a single-leaf chunk
                assert len(cdata) == 1 or size <= chunker.chunk_chars * 2


# ---------------------------------------------------------------------------
# Output validation — length, valid JSON, merge equivalence
# ---------------------------------------------------------------------------


class TestOutputValidation:
    """Comprehensive contract tests: every chunk must be valid JSON,
    output text length must respect chunk_chars, and merging all chunks
    must reconstruct the original data."""

    @pytest.mark.parametrize(
        "data, chunk_chars",
        [
            # Flat object
            ({f"k{i}": f"val_{i}" * 10 for i in range(30)}, 300),
            # Nested objects
            (
                {"a": {"b": {"c": "x" * 100}}, "d": {"e": "y" * 100}, "f": {"g": "z" * 100}},
                200,
            ),
            # Array of objects
            ([{"id": i, "data": "x" * 50} for i in range(20)], 300),
            # Mixed nesting
            (
                {
                    "users": [{"name": f"u{i}", "tags": [f"t{j}" for j in range(3)]} for i in range(10)],
                    "meta": {"version": "1.0"},
                },
                400,
            ),
            # Deep nesting with sibling
            (
                {"l1": {"l2": {"l3": {"l4": {"v": "deep" * 50}}}, "sibling": "x" * 100}},
                256,
            ),
            # CJK content
            ({f"键{i}": f"中文值_{i}" * 10 for i in range(20)}, 300),
            # Flat array of scalars
            (list(range(200)), 200),
            # Object with long string values
            ({f"k{i}": "x" * 200 for i in range(10)}, 500),
            # Large nested array
            ([{"name": f"item_{i}", "vals": list(range(10))} for i in range(30)], 400),
        ],
        ids=[
            "flat_object",
            "nested_objects",
            "array_of_objects",
            "mixed_nesting",
            "deep_nesting",
            "cjk",
            "flat_array",
            "long_values",
            "nested_array",
        ],
    )
    def test_output_contract(self, make_json, data, chunk_chars):
        chunker = JsonFileChunker(chunk_chars=chunk_chars)
        path = make_json(data=data)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) >= 1
        limit = chunker.chunk_chars  # accounts for clamping (min 256)

        # 1. Every chunk text must be valid JSON
        parsed_chunks = []
        for i, c in enumerate(chunks):
            try:
                parsed_chunks.append(json.loads(c.text))
            except json.JSONDecodeError as exc:
                pytest.fail(f"Chunk {i} is not valid JSON: {exc}")

        # 2. Output text length must be <= chunk_chars (actual, after clamp)
        #    Single-leaf chunks may exceed — that's acceptable.
        for i, c in enumerate(chunks):
            if len(c.text) > limit:
                assert _is_single_leaf(parsed_chunks[i]), (
                    f"Chunk {i} text length {len(c.text)} > limit={limit} " f"but is not a single leaf"
                )

        # 3. All leaf values from chunks must reconstruct the original data
        #    (works for both dict-rooted and array-rooted JSON)
        original_leaves = list(_all_leaf_values(data))
        chunk_leaves = []
        for p in parsed_chunks:
            chunk_leaves.extend(_all_leaf_values(p))
        assert chunk_leaves == original_leaves, (
            f"Leaf values mismatch: expected {len(original_leaves)} leaves, " f"got {len(chunk_leaves)}"
        )

    def test_length_exact_match_sizenode(self, make_json):
        """Output text length should exactly equal the compact JSON size."""
        data = {f"key_{i}": f"val_{i}_" + "x" * 50 for i in range(50)}
        chunk_chars = 500
        path = make_json(data=data)
        _, chunks = _run(JsonFileChunker(chunk_chars=chunk_chars).chunk(path))
        assert len(chunks) > 1
        for c in chunks:
            parsed = json.loads(c.text)
            expected_len = len(json.dumps(parsed, ensure_ascii=False))
            assert len(c.text) == expected_len, f"Text length {len(c.text)} != compact json size {expected_len}"


# ---------------------------------------------------------------------------
# FileChunk / FileNode properties
# ---------------------------------------------------------------------------


class TestFileProperties:
    def test_hash_id_deterministic(self, make_json):
        path = make_json(data={"hello": "world"})
        _, chunks = _run(JsonFileChunker().chunk(path))
        assert len(chunks) == 1
        assert chunks[0].id
        _, chunks2 = _run(JsonFileChunker().chunk(path))
        assert chunks[0].id == chunks2[0].id

    def test_node_properties(self, make_json):
        path = make_json(data={"a": 1})
        node, chunks = _run(JsonFileChunker().chunk(path))
        assert node.st_mtime > 0
        assert node.links == []
        assert node.chunk_ids == [c.id for c in chunks]
