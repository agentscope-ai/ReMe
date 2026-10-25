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


def segments(path: Path, encoding: str, budget):
    """Keep line/record boundaries where possible; split long records without losing text."""
    pending = ""
    offset = 0
    limit = budget.limits["segment"]
    with path.open(encoding=encoding, errors="strict", newline="") as stream:
        for line in stream:
            if pending and budget.count(pending + line) > limit:
                yield {"offset": offset, "text": pending}
                offset += len(pending)
                pending = ""
            while budget.count(line) > limit:
                end = budget.prefix(line, limit)
                yield {"offset": offset, "text": line[:end]}
                offset += end
                line = line[end:]
            pending += line
        if pending:
            yield {"offset": offset, "text": pending}
