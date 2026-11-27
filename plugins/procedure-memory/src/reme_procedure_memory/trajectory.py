"""Local trajectory import and bounded, lossless text segmentation."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from reme.steps.file_io import validate_filename_component
from reme.steps.file_io._file_io import detect_file_encoding

from .checkpoint import atomic_json, checked_path, digest, load_json

SUPPORTED = {"", ".txt", ".md", ".markdown", ".json", ".jsonl", ".log"}
SEGMENTATION_VERSION = 5
_MAX_SEGMENT_CHARS = 1024 * 1024
_DENSITY_SAMPLE_CHARS = 4096


def file_hash(path: Path, max_bytes: int) -> str:
    """Stream a bounded source without loading it into memory."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if size > max_bytes:
                raise ValueError(f"Trajectory exceeds max_file_bytes={max_bytes}")
            hasher.update(block)
    return hasher.hexdigest()


def import_file(workspace: Path, root: Path, metadata: Path, file: str, filename: str, request_id: str, max_bytes: int):
    """Copy bytes once; journal the allocated name before publishing the complete copy.

    The owning component serializes import calls within the supported single process.
    """
    if error := validate_filename_component(filename):
        raise ValueError(error)
    source = Path(file).expanduser().resolve()
    record_path = metadata / "imports" / f"{digest(request_id)}.json"
    if record_path.exists():
        record = load_json(record_path)
        saved_input = record["input"]
        if saved_input["file"] != str(source) or saved_input["filename"] != filename:
            raise ValueError("request_id already belongs to a different input")
        target = checked_path(workspace, record["source"], root)
        if target.exists():
            if file_hash(target, max_bytes) != saved_input["sha256"]:
                raise ValueError("Saved trajectory differs from the import record")
            # A published copy is sufficient to resume, even if the original
            # has gone. Retain conflict detection when it is still available.
            if source.exists():
                if not source.is_file():
                    raise ValueError("file must name a local regular file")
                if file_hash(source, max_bytes) != saved_input["sha256"]:
                    raise ValueError("request_id already belongs to a different input")
            record["complete"] = True
            atomic_json(record_path, record)
            return record

    # The external input is required only until a complete copy is published.
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("file must name a local regular file")
    checksum = file_hash(source, max_bytes)
    identity = {"file": str(source), "filename": filename, "sha256": checksum}
    if record_path.exists():
        if record["input"] != identity:
            raise ValueError("request_id already belongs to a different input")
    else:
        # An interrupted copy has already promised its name to that request.
        # A different import must not take it before the original request resumes.
        reserved = {load_json(path)["source"] for path in (metadata / "imports").glob("*.json")}
        candidate = filename
        number = 2
        while True:
            target = checked_path(workspace, (root / candidate).relative_to(workspace).as_posix(), root)
            if (
                not target.exists()
                and not target.is_symlink()
                and target.relative_to(workspace).as_posix() not in reserved
            ):
                break
            original = Path(filename)
            candidate = f"{original.stem}-{number}{original.suffix}"
            number += 1
        record = {"input": identity, "source": target.relative_to(workspace).as_posix(), "complete": False}
        atomic_json(record_path, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".trajectory-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if file_hash(Path(temporary), max_bytes) != checksum:
            raise ValueError("Input changed during copy; retry with a stable file")
        os.link(temporary, target)  # Atomic publication that cannot overwrite an existing trajectory.
    finally:
        Path(temporary).unlink(missing_ok=True)
    record["complete"] = True
    atomic_json(record_path, record)
    return record


async def prepare_text(path: Path) -> str:
    """Validate supported formats before extracting any experiences; return encoding."""
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported trajectory format: {path.suffix}; original file is retained")
    encoding = await detect_file_encoding(path)
    with path.open(encoding=encoding, errors="strict", newline="") as stream:
        if path.suffix.lower() == ".json":
            json.load(stream)
        else:
            for number, line in enumerate(stream, 1):
                if "\x00" in line:
                    raise ValueError("Binary trajectory is unsupported; original file is retained")
                if path.suffix.lower() == ".jsonl" and line.strip():
                    try:
                        json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"Invalid JSONL record on line {number}") from exc
    return encoding


def _segment_end(text: str, start: int, length: int) -> int:
    """Prefer a line ending in the latter half without splitting CRLF when possible."""
    end = start + length
    if length > 1 and text[end - 1 : end + 1] == "\r\n":
        end -= 1
    lower = start + (length - 1) // 2
    boundary = max(text.rfind("\n", lower, end), text.rfind("\r", lower, end))
    return boundary + 1 if boundary >= lower else end


def segments(path: Path, encoding: str, budget):
    """Predict bounded candidates, check exactly once, and halve oversized candidates.

    Probes shrink geometrically; the next prediction grows by at most twice the
    emitted length. Including one initial density sample, total counter input is
    less than 8N characters, even for nonmonotonic counts. Segments need not fill
    the budget.
    """
    limit = budget.limits["segment"]
    target_tokens = limit - limit // 20
    target = min(limit, _MAX_SEGMENT_CHARS, _DENSITY_SAMPLE_CHARS)
    calibrating = True
    pending = ""
    start = offset = 0
    eof = False
    with path.open(encoding=encoding, errors="strict", newline="") as stream:
        while True:
            # One character of lookahead keeps CRLF intact at window edges.
            needed = target + 1 - (len(pending) - start)
            if needed > 0 and not eof:
                incoming = stream.read(needed)
                pending = pending[start:] + incoming
                start = 0
                eof = len(incoming) < needed
            if start == len(pending):
                break

            length = min(target, len(pending) - start)
            while True:
                end = start + length if eof and length == len(pending) - start else _segment_end(pending, start, length)
                text = pending[start:end]
                count = budget.count(text)
                if count <= limit:
                    break
                length = len(text) // 2
                if not length:
                    raise ValueError("Token budget cannot hold one character")
            predicted = min(_MAX_SEGMENT_CHARS, max(1, len(text) * target_tokens // max(1, count)))
            if calibrating:
                calibrating = False
                # Calibrate once before emitting so a sparse first sample does
                # not become an unnecessary model invocation of its own.
                if predicted > len(text) and (not eof or end < len(pending)):
                    target = predicted
                    continue
            yield {"offset": offset, "text": text}
            offset += len(text)
            start = end
            # Density is only a scheduling hint. Capping growth also bounds the
            # cost of mispredictions when token density changes abruptly.
            target = min(predicted, len(text) * 2)
