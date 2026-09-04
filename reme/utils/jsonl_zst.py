"""Tiny JSONL-over-zstd helpers."""

import io
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from uuid import uuid4

import zstandard as zstd


def read_jsonl_zst(path: str | Path, encoding: str = "utf-8") -> Iterator[str]:
    """Read JSONL-over-zstd lines from a file."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            text = io.TextIOWrapper(reader, encoding=encoding)
            yield from text


def write_jsonl_zst(path: str | Path, lines: Iterable[str], encoding: str = "utf-8") -> Path:
    """Write JSONL-over-zstd lines to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with tmp.open("wb") as raw:
        with zstd.ZstdCompressor(level=3).stream_writer(raw) as writer:
            text = io.TextIOWrapper(writer, encoding=encoding)
            for line in lines:
                text.write(line)
                if not line.endswith("\n"):
                    text.write("\n")
            text.flush()
            text.detach()
    os.replace(tmp, path)
    return path
