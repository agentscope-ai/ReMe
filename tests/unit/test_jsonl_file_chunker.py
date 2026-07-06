"""Tests for JsonlFileChunker."""

# pylint: disable=protected-access

import asyncio
import json
import os
import tempfile

from reme.components.file_chunker import JsonlFileChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _write_jsonl(lines: list[str]) -> str:
    """Write lines to a temp .jsonl file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
    return path


def _make_records(n: int, width: int = 20) -> list[str]:
    """Generate *n* JSONL lines, each roughly *width* chars of JSON."""
    return [json.dumps({"id": i, "text": "x" * max(0, width - 20)}) for i in range(n)]


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_empty_file():
    """Empty file → zero chunks."""
    path = _write_jsonl([])
    try:
        chunker = JsonlFileChunker()
        node, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 0
        assert node.links == []
        print("✓ test_empty_file passed")
    finally:
        os.unlink(path)


def test_blank_lines_only():
    """File with only blank lines → zero chunks."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        f.write("\n\n\n")
    try:
        chunker = JsonlFileChunker()
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 0
        print("✓ test_blank_lines_only passed")
    finally:
        os.unlink(path)


def test_single_line():
    """One line → one chunk."""
    lines = _make_records(1)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker()
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1
        assert lines[0] in chunks[0].text
        print("✓ test_single_line passed")
    finally:
        os.unlink(path)


def test_all_lines_fit_in_one_chunk():
    """Small file fits entirely in one chunk."""
    lines = _make_records(5, width=30)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=5000)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 5
        print("✓ test_all_lines_fit_in_one_chunk passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Multi-chunk splitting (no overlap)
# ---------------------------------------------------------------------------


def test_multiple_chunks_no_overlap():
    """Large file is split into multiple chunks with no overlap."""
    # Each line ≈ 32 chars (with \n), 10 lines ≈ 320 chars.
    lines = _make_records(10, width=30)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=100, max_overlap_chars=0)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) > 1, f"Expected >1 chunks, got {len(chunks)}"
        # No overlap: consecutive chunks don't share lines.
        for i in range(len(chunks) - 1):
            assert chunks[i].end_line < chunks[i + 1].start_line
        # All lines are covered.
        assert chunks[0].start_line == 1
        assert chunks[-1].end_line == 10
        print(f"  Created {len(chunks)} chunks")
        print("✓ test_multiple_chunks_no_overlap passed")
    finally:
        os.unlink(path)


def test_line_aligned_no_intra_line_split():
    """Every chunk boundary falls on a line boundary."""
    lines = _make_records(20, width=40)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=150, max_overlap_chars=0)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) > 1
        for c in chunks:
            # Each chunk text should be a concatenation of whole lines.
            text_lines = c.text.splitlines()
            assert len(text_lines) == (c.end_line - c.start_line + 1)
        print("✓ test_line_aligned_no_intra_line_split passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Overlap behaviour
# ---------------------------------------------------------------------------


def test_overlap_lines_shared():
    """Overlap lines appear at the end of chunk N and the start of chunk N+1."""
    # Each line is ~32 chars.  max_chars=100 → ~3 lines per chunk.
    # max_overlap_chars=40 → ~1 line of overlap.
    lines = _make_records(12, width=30)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=100, max_overlap_chars=40)
        _, chunks = _run(chunker.chunk(path))
        assert len(chunks) > 1
        # Verify overlap: chunk[i].end_line >= chunk[i+1].start_line
        for i in range(len(chunks) - 1):
            assert chunks[i].end_line >= chunks[i + 1].start_line, (
                f"Chunk {i} end_line={chunks[i].end_line} < " f"chunk {i+1} start_line={chunks[i+1].start_line}"
            )
        # Last chunk still reaches the end of the file.
        assert chunks[-1].end_line == 12
        print(f"  Created {len(chunks)} chunks with overlap")
        print("✓ test_overlap_lines_shared passed")
    finally:
        os.unlink(path)


def test_zero_overlap():
    """max_overlap_chars=0 produces strictly non-overlapping chunks."""
    lines = _make_records(10, width=30)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=100, max_overlap_chars=0)
        _, chunks = _run(chunker.chunk(path))
        for i in range(len(chunks) - 1):
            assert chunks[i].end_line < chunks[i + 1].start_line
        print("✓ test_zero_overlap passed")
    finally:
        os.unlink(path)


def test_overlap_capped_by_max_overlap_chars():
    """Overlap never exceeds max_overlap_chars in size."""
    lines = _make_records(10, width=50)
    path = _write_jsonl(lines)
    try:
        max_overlap = 60  # Should allow at most 1 line of overlap (~51 chars).
        chunker = JsonlFileChunker(max_chars=200, max_overlap_chars=max_overlap)
        _, chunks = _run(chunker.chunk(path))
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                overlap_start = chunks[i + 1].start_line
                overlap_end = chunks[i].end_line
                if overlap_start <= overlap_end:
                    overlap_lines = list(range(overlap_start, overlap_end + 1))
                    overlap_text = "".join(lines[ln - 1] for ln in overlap_lines)
                    assert len(overlap_text) <= max_overlap, f"Overlap size {len(overlap_text)} > {max_overlap}"
        print("✓ test_overlap_capped_by_max_overlap_chars passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Oversized single line
# ---------------------------------------------------------------------------


def test_single_line_exceeds_max():
    """A line larger than max_chars is emitted alone (no intra-line split)."""
    long_line = json.dumps({"data": "A" * 200})
    lines = ["short", long_line, "short"]
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=64, max_overlap_chars=0)
        _, chunks = _run(chunker.chunk(path))
        # The long line must appear in exactly one chunk by itself (or with
        # no split).
        found = False
        for c in chunks:
            if long_line in c.text:
                found = True
                break
        assert found, "Long line was not found in any chunk"
        assert len(chunks) >= 2  # At least some splitting happened.
        print("✓ test_single_line_exceeds_max passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Byte mode
# ---------------------------------------------------------------------------


def test_byte_mode():
    """mode='bytes' counts byte length instead of char length."""
    # CJK characters are 3 bytes each in UTF-8 but 1 char each.
    # Use ensure_ascii=False so raw CJK chars appear in the file.
    cjk_record = json.dumps({"text": "你" * 10}, ensure_ascii=False)
    # The JSON string itself has CJK chars; after writing to file each line
    # has ~13 ASCII chars ({"text": "..."}) + 10 CJK chars.
    # char length ≈ 23, byte length ≈ 13 + 10*3 = 43.
    lines = [cjk_record] * 6
    path = _write_jsonl(lines)
    try:
        # In char mode: each line ≈ 24 chars → 2 lines per chunk at max=50.
        chunker_chars = JsonlFileChunker(max_chars=50, max_overlap_chars=0, mode="chars")
        _, chunks_chars = _run(chunker_chars.chunk(path))

        # In byte mode: each line ≈ 44 bytes → 1 line per chunk at max=50.
        chunker_bytes = JsonlFileChunker(max_chars=50, max_overlap_chars=0, mode="bytes")
        _, chunks_bytes = _run(chunker_bytes.chunk(path))

        # Byte mode should produce more chunks since each line is larger in bytes.
        assert len(chunks_bytes) > len(chunks_chars), (
            f"Byte mode ({len(chunks_bytes)}) should produce more chunks "
            f"than char mode ({len(chunks_chars)}) for CJK text"
        )
        print(f"  char mode: {len(chunks_chars)} chunks, byte mode: {len(chunks_bytes)} chunks")
        print("✓ test_byte_mode passed")
    finally:
        os.unlink(path)


def test_invalid_mode_raises():
    """An invalid mode string raises ValueError."""
    try:
        JsonlFileChunker(mode="words")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✓ test_invalid_mode_raises passed")


# ---------------------------------------------------------------------------
# FileNode / FileChunk properties
# ---------------------------------------------------------------------------


def test_file_chunk_properties():
    """Each FileChunk has correct path, line range, text, and hash id."""
    lines = _make_records(5, width=30)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=5000)
        _, chunks = _run(chunker.chunk(path))
        c = chunks[0]
        assert c.start_line >= 1
        assert c.end_line >= c.start_line
        assert c.id and len(c.id) > 0
        assert c.text.strip()
        print("✓ test_file_chunk_properties passed")
    finally:
        os.unlink(path)


def test_file_node_properties():
    """FileNode carries path, st_mtime, chunk_ids, and empty links."""
    lines = _make_records(3)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker()
        node, chunks = _run(chunker.chunk(path))
        assert node.st_mtime > 0
        assert node.links == []
        assert node.chunk_ids == [c.id for c in chunks]
        print("✓ test_file_node_properties passed")
    finally:
        os.unlink(path)


def test_hash_id_deterministic():
    """Same file produces the same chunk hash ids on repeated calls."""
    lines = _make_records(5, width=30)
    path = _write_jsonl(lines)
    try:
        c1 = JsonlFileChunker(max_chars=100)
        _, chunks1 = _run(c1.chunk(path))
        c2 = JsonlFileChunker(max_chars=100)
        _, chunks2 = _run(c2.chunk(path))
        assert [c.id for c in chunks1] == [c.id for c in chunks2]
        print("✓ test_hash_id_deterministic passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_max_chars_floor():
    """max_chars is clamped to at least 64."""
    c = JsonlFileChunker(max_chars=1)
    assert c.max_chars == 64
    print("✓ test_max_chars_floor passed")


def test_max_overlap_floor():
    """Negative max_overlap_chars is clamped to 0."""
    c = JsonlFileChunker(max_overlap_chars=-10)
    assert c.max_overlap_chars == 0
    print("✓ test_max_overlap_floor passed")


def test_full_coverage():
    """Every line of the file appears in at least one chunk."""
    lines = _make_records(25, width=40)
    path = _write_jsonl(lines)
    try:
        chunker = JsonlFileChunker(max_chars=150, max_overlap_chars=50)
        _, chunks = _run(chunker.chunk(path))
        covered: set[int] = set()
        for c in chunks:
            covered.update(range(c.start_line, c.end_line + 1))
        assert covered == set(range(1, 26)), f"Missing lines: {set(range(1, 26)) - covered}"
        print("✓ test_full_coverage passed")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Internal algorithm
# ---------------------------------------------------------------------------


def test_chunk_lines_algorithm_simple():
    """_chunk_lines produces correct ranges for a simple case."""
    # max_chars floor is 64, so lines must be long enough.
    # Each line = 20 chars + \n = 21 chars.  64 / 21 = 3 lines per chunk.
    chunker = JsonlFileChunker(max_chars=64, max_overlap_chars=0)
    lines = ["a" * 20 + "\n", "b" * 20 + "\n", "c" * 20 + "\n", "d" * 20 + "\n"]
    # 3 lines = 63 chars ≤ 64 → chunk 1: (0,3).  4th line alone → chunk 2: (3,4).
    ranges = chunker._chunk_lines(lines)
    assert ranges == [(0, 3), (3, 4)], f"Got {ranges}"
    print("✓ test_chunk_lines_algorithm_simple passed")


def test_chunk_lines_algorithm_with_overlap():
    """_chunk_lines with overlap produces overlapping ranges."""
    # Each line = 20 chars + \n = 21 chars.  max_chars=64 → 3 lines per chunk.
    # max_overlap_chars=22 → 1 line of overlap (21 chars ≤ 22).
    chunker = JsonlFileChunker(max_chars=64, max_overlap_chars=22)
    lines = ["a" * 20 + "\n"] * 6
    ranges = chunker._chunk_lines(lines)
    assert len(ranges) >= 2, f"Expected ≥2 ranges, got {ranges}"
    # Verify overlap: range[i].end >= range[i+1].start
    for i in range(len(ranges) - 1):
        assert ranges[i][1] >= ranges[i + 1][0], f"No overlap between range {i} {ranges[i]} and {i+1} {ranges[i+1]}"
    # Last range reaches end.
    assert ranges[-1][1] == 6
    print("✓ test_chunk_lines_algorithm_with_overlap passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_empty_file()
    test_blank_lines_only()
    test_single_line()
    test_all_lines_fit_in_one_chunk()
    test_multiple_chunks_no_overlap()
    test_line_aligned_no_intra_line_split()
    test_overlap_lines_shared()
    test_zero_overlap()
    test_overlap_capped_by_max_overlap_chars()
    test_single_line_exceeds_max()
    test_byte_mode()
    test_invalid_mode_raises()
    test_file_chunk_properties()
    test_file_node_properties()
    test_hash_id_deterministic()
    test_max_chars_floor()
    test_max_overlap_floor()
    test_full_coverage()
    test_chunk_lines_algorithm_simple()
    test_chunk_lines_algorithm_with_overlap()
    print("\n所有测试通过!")
