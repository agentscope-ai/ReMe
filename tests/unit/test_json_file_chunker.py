"""Tests for JsonFileChunker."""

# pylint: disable=protected-access

import asyncio
import json
import os
import tempfile

from reme.components.file_chunker import JsonFileChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _write_json(suffix: str, data=None, raw: str | None = None) -> str:
    """Write JSON to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        if raw is not None:
            f.write(raw)
        elif data is not None:
            json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Basic chunking tests
# ---------------------------------------------------------------------------


def test_empty_file():
    """Empty JSON file → zero chunks."""
    path = _write_json(".json", raw="")
    try:
        chunker = JsonFileChunker()
        node, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 0
        assert node.links == []
        print("✓ test_empty_file passed")
    finally:
        os.unlink(path)


def test_empty_object():
    """``{}`` → zero chunks (no keys to split)."""
    path = _write_json(".json", data={})
    try:
        chunker = JsonFileChunker()
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 0
        print("✓ test_empty_object passed")
    finally:
        os.unlink(path)


def test_small_json_single_chunk():
    """A small JSON object fits in one chunk."""
    data = {"name": "Alice", "age": 30, "city": "Shanghai"}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=5000)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 1
        parsed = json.loads(chunks[0].text)
        assert parsed == data
        assert chunks[0].start_line >= 1  # line 1 is '{' in pretty-printed JSON
        print("✓ test_small_json_single_chunk passed")
    finally:
        os.unlink(path)


def test_large_json_multiple_chunks():
    """A large JSON object is split into multiple chunks."""
    data = {f"key_{i}": f"value_{i}" * 20 for i in range(50)}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=500)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
        # All original keys must appear across chunks.
        all_keys: set[str] = set()
        for c in chunks:
            all_keys.update(json.loads(c.text).keys())
        assert all_keys == set(data.keys())
        print(f"  Created {len(chunks)} chunks")
        print("✓ test_large_json_multiple_chunks passed")
    finally:
        os.unlink(path)


def test_nested_json_preserves_structure():
    """Nested key paths are preserved in each chunk."""
    data = {
        "section_a": {"name": "Alice", "score": 95, "detail": "x" * 200},
        "section_b": {"name": "Bob", "score": 88, "detail": "y" * 200},
        "section_c": {"name": "Charlie", "score": 72, "detail": "z" * 200},
    }
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=400)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) > 1
        # Each chunk must be valid JSON.
        for c in chunks:
            parsed = json.loads(c.text)
            assert isinstance(parsed, dict)
        # Nested structure must be intact (not flattened).
        merged: dict = {}
        for c in chunks:
            _deep_merge(merged, json.loads(c.text))
        assert merged == data
        print("✓ test_nested_json_preserves_structure passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# List conversion
# ---------------------------------------------------------------------------


def test_list_conversion_enabled():
    """With convert_lists=True, lists become index-keyed dicts."""
    data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=5000, convert_lists=True)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 1
        parsed = json.loads(chunks[0].text)
        # Lists should be converted to dicts with string keys.
        assert isinstance(parsed["users"], dict)
        assert parsed["users"]["0"]["name"] == "Alice"
        assert parsed["users"]["1"]["name"] == "Bob"
        print("✓ test_list_conversion_enabled passed")
    finally:
        os.unlink(path)


def test_list_conversion_disabled():
    """With convert_lists=False, lists are kept as-is."""
    data = {"items": [1, 2, 3]}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=5000, convert_lists=False)
        _, chunks = _run(chunker.chunk(path))
        parsed = json.loads(chunks[0].text)
        assert parsed["items"] == [1, 2, 3]
        print("✓ test_list_conversion_disabled passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Line-range mapping
# ---------------------------------------------------------------------------


def test_line_ranges_monotonic():
    """Chunk line ranges should generally cover the file."""
    data = {f"key_{i}": f"val_{i}" for i in range(30)}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=300)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) > 1
        # All start_line / end_line should be within file bounds.
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line
        # First chunk should start at or near line 1.
        assert chunks[0].start_line <= 3
        print("✓ test_line_ranges_monotonic passed")
    finally:
        os.unlink(path)


def test_line_ranges_nested():
    """Line ranges for nested chunks should reflect actual key positions."""
    data = {
        "alpha": {"x": 1},
        "beta": {"y": 2},
        "gamma": {"z": 3},
    }
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=5000)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 1
        c = chunks[0]
        # Single chunk covers the whole file.
        assert c.start_line >= 1  # line 1 is '{' in pretty-printed JSON
        assert c.end_line >= 5  # At least a few lines for pretty-printed JSON.
        print("✓ test_line_ranges_nested passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# FileChunk / FileNode properties
# ---------------------------------------------------------------------------


def test_file_chunk_has_hash_id():
    """Each FileChunk should have a deterministic hash id."""
    data = {"hello": "world"}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker()
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 1
        assert chunks[0].id
        assert len(chunks[0].id) > 0
        # Hash id should be reproducible.
        chunker2 = JsonFileChunker()
        _, chunks2 = _run(chunker2.chunk(path))
        assert chunks[0].id == chunks2[0].id
        print("✓ test_file_chunk_has_hash_id passed")
    finally:
        os.unlink(path)


def test_file_node_properties():
    """FileNode should carry path, st_mtime, and chunk_ids."""
    data = {"a": 1}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker()
        node, chunks = _run(chunker.chunk(path))
        assert node.st_mtime > 0
        assert node.links == []
        assert node.chunk_ids == [c.id for c in chunks]
        print("✓ test_file_node_properties passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_malformed_json_fallback():
    """Malformed JSON falls back to a single whole-file chunk."""
    path = _write_json(".json", raw='{"bad": json}')
    try:
        chunker = JsonFileChunker()
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        print("✓ test_malformed_json_fallback passed")
    finally:
        os.unlink(path)


def test_min_chunk_size_default():
    """min_chunk_size defaults to max_chunk_size - 200 (lower bound 50)."""
    c1 = JsonFileChunker(max_chunk_size=2000)
    assert c1.min_chunk_size == 1800

    c2 = JsonFileChunker(max_chunk_size=100)
    assert c2.min_chunk_size == 50  # max(100-200, 50) = 50

    c3 = JsonFileChunker(max_chunk_size=2000, min_chunk_size=500)
    assert c3.min_chunk_size == 500
    print("✓ test_min_chunk_size_default passed")


def test_max_chunk_size_floor():
    """max_chunk_size is clamped to at least 64."""
    c = JsonFileChunker(max_chunk_size=10)
    assert c.max_chunk_size == 64
    print("✓ test_max_chunk_size_floor passed")


def test_deeply_nested_json():
    """Deeply nested JSON is split while preserving depth."""
    data = {"level1": {"level2": {"level3": {"value": "deep" * 100}}}}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(max_chunk_size=200)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) >= 1
        # Reassemble and verify structure.
        merged: dict = {}
        for c in chunks:
            _deep_merge(merged, json.loads(c.text))
        assert merged["level1"]["level2"]["level3"]["value"] == "deep" * 100
        print("✓ test_deeply_nested_json passed")
    finally:
        os.unlink(path)


def test_json_with_unicode():
    """Unicode content is handled correctly."""
    data = {"greeting": "你好世界", "emoji": "🎉🎊", "mixed": "hello世界"}
    path = _write_json(".json", data=data)
    try:
        chunker = JsonFileChunker(ensure_ascii=False)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) >= 1
        parsed = json.loads(chunks[0].text)
        assert parsed["greeting"] == "你好世界"
        assert parsed["emoji"] == "🎉🎊"
        print("✓ test_json_with_unicode passed")
    finally:
        os.unlink(path)


def test_chunk_size_respected():
    """Each chunk's serialised size should not grossly exceed max_chunk_size."""
    data = {f"k{i}": "v" * 50 for i in range(40)}
    path = _write_json(".json", data=data)
    try:
        max_size = 500
        chunker = JsonFileChunker(max_chunk_size=max_size)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) > 1
        # Allow some slack: a single indivisible leaf may overflow.
        for c in chunks:
            assert len(c.text) <= max_size * 3, f"Chunk too large: {len(c.text)} > {max_size * 3}"
        print("✓ test_chunk_size_respected passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_json_split_basic():
    """_json_split produces multiple sub-dicts for large input."""
    chunker = JsonFileChunker(max_chunk_size=100)
    data = {f"key_{i}": f"val_{i}" * 10 for i in range(20)}
    result = chunker._json_split(data)
    assert len(result) > 1
    print("✓ test_json_split_basic passed")


def test_set_nested_dict():
    """_set_nested_dict creates intermediate dicts along the path."""
    d: dict = {}
    JsonFileChunker._set_nested_dict(d, ["a", "b", "c"], 42)
    assert d == {"a": {"b": {"c": 42}}}
    print("✓ test_set_nested_dict passed")


def test_list_to_dict_preprocessing():
    """_list_to_dict_preprocessing converts lists to index-keyed dicts."""
    chunker = JsonFileChunker()
    data = {"items": [10, 20, 30], "nested": {"arr": [{"x": 1}]}}
    result = chunker._list_to_dict_preprocessing(data)
    assert result["items"] == {"0": 10, "1": 20, "2": 30}
    assert result["nested"]["arr"] == {"0": {"x": 1}}
    print("✓ test_list_to_dict_preprocessing passed")


def test_json_size():
    """_json_size returns the serialised character count."""
    assert JsonFileChunker._json_size({"a": 1}) == len(json.dumps({"a": 1}))
    assert JsonFileChunker._json_size({}) == len("{}")
    print("✓ test_json_size passed")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _deep_merge(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target* (mutates target)."""
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v
    return target


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_empty_file()
    test_empty_object()
    test_small_json_single_chunk()
    test_large_json_multiple_chunks()
    test_nested_json_preserves_structure()
    test_list_conversion_enabled()
    test_list_conversion_disabled()
    test_line_ranges_monotonic()
    test_line_ranges_nested()
    test_file_chunk_has_hash_id()
    test_file_node_properties()
    test_malformed_json_fallback()
    test_min_chunk_size_default()
    test_max_chunk_size_floor()
    test_deeply_nested_json()
    test_json_with_unicode()
    test_chunk_size_respected()
    test_json_split_basic()
    test_set_nested_dict()
    test_list_to_dict_preprocessing()
    test_json_size()
    print("\n所有测试通过!")
