"""Original-format imports, filename collisions and lossless segmentation."""

# pylint: disable=missing-function-docstring

import pytest

from reme_procedure_memory.checkpoint import atomic_json, digest, load_json
from reme_procedure_memory.token_budget import TokenBudget
from reme_procedure_memory.trajectory import prepare_text, segments


@pytest.mark.parametrize("size", [1024 * 1024, 4 * 1024 * 1024])
@pytest.mark.parametrize("chars_per_token", [1, 4])
def test_long_line_tokenization_is_bounded_by_segment_size(tmp_path, size, chars_per_token):
    counted = 0
    largest = 0

    def count(text):
        nonlocal counted, largest
        counted += len(text)
        largest = max(largest, len(text))
        return (len(text) + chars_per_token - 1) // chars_per_token

    text = "x" * size + "\r\nlast line\r\n"
    path = tmp_path / "long.log"
    path.write_bytes(text.encode())
    budget = TokenBudget(counter=count)
    chunks = list(segments(path, "utf-8", budget))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(0 < len(chunk["text"]) <= budget.limits["segment"] * chars_per_token for chunk in chunks)
    assert [chunk["offset"] for chunk in chunks] == [
        sum(len(previous["text"]) for previous in chunks[:n]) for n in range(len(chunks))
    ]
    assert largest <= budget.limits["segment"] * chars_per_token + 1
    assert counted < len(text) * 8


def test_multibyte_long_lines_remain_lossless(tmp_path):
    text = "leading\r\n" + "你好🙂" * 1000 + "\r\ntrailing\n"
    path = tmp_path / "unicode.log"
    path.write_bytes(text.encode())
    budget = TokenBudget(counter=lambda value: len(value.encode()), segment=63)
    chunks = list(segments(path, "utf-8", budget))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(len(chunk["text"].encode()) <= 63 for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,text",
    [
        ("a.txt", "你好 world\r\n"),
        ("a.md", "# Task\n**repair**\n"),
        ("a.json", '{"reasoning": "全部", "other": [1,2]}'),
        ("a.jsonl", '{"first":1}\n{"second":2}\n'),
        ("a.log", "[INFO] all text\n"),
        ("no-extension", "plain text"),
    ],
)
async def test_formats_preserve_bytes(runtime, tmp_path, filename, text):
    _, store, _ = runtime
    source = tmp_path / filename
    source.write_bytes(text.encode("utf-8"))
    imported = store.import_trajectory(str(source), filename, filename)
    path = store.source_path(imported["source"])
    assert path.read_bytes() == source.read_bytes()
    encoding = await prepare_text(path)
    budget = TokenBudget(counter=len, segment=8)
    chunks = list(segments(path, encoding, budget))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(len(chunk["text"]) <= 8 for chunk in chunks)
    assert [chunk["offset"] for chunk in chunks] == [
        sum(len(c["text"]) for c in chunks[:n]) for n in range(len(chunks))
    ]


@pytest.mark.asyncio
async def test_collision_rules_and_idempotency(runtime, tmp_path):
    _, store, _ = runtime
    source = tmp_path / "run.jsonl"
    source.write_text("{}\n")
    for number in range(1, 4):
        result = store.import_trajectory(str(source), "run.jsonl", str(number))
        suffix = "" if number == 1 else f"-{number}"
        assert result["source"] == f"session/trajectories/run{suffix}.jsonl"
        assert store.import_trajectory(str(source), "run.jsonl", str(number)) == result
    result = store.import_trajectory(str(source), "run-2.jsonl", "numbered")
    assert result["source"].endswith("run-2-2.jsonl")
    store.import_trajectory(str(source), "run", "no-ext-1")
    assert store.import_trajectory(str(source), "run", "no-ext-2")["source"].endswith("/run-2")
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="different input"):
        store.import_trajectory(str(source), "run.jsonl", "1")


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False])
async def test_resume_published_copy_without_original(runtime, tmp_path, complete):
    _, store, _ = runtime
    source = tmp_path / "original.log"
    source.write_bytes(b"original\r\ntrajectory")
    record = store.import_trajectory(str(source), "saved.log", "saved-copy")
    record_path = store.state_root / "imports" / f"{digest('saved-copy')}.json"
    if not complete:
        # Simulate publication succeeding before the import journal was finalized.
        record["complete"] = False
        atomic_json(record_path, record)
    source.unlink()

    resumed = store.import_trajectory(str(source), "saved.log", "saved-copy")
    assert resumed == {**record, "complete": True}
    assert load_json(record_path) == resumed
    assert store.source_path(resumed["source"]).read_bytes() == b"original\r\ntrajectory"
    assert not (store.trajectory_root / "saved-2.log").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["filename", "path", "copy"])
async def test_resume_without_original_validates_identity_and_copy(runtime, tmp_path, change):
    _, store, _ = runtime
    source = tmp_path / "original.log"
    source.write_text("evidence")
    record = store.import_trajectory(str(source), "saved.log", "identity")
    source.unlink()
    filename = "saved.log"
    if change == "filename":
        filename = "different.log"
    elif change == "path":
        source = tmp_path / "different.log"
    else:
        store.source_path(record["source"]).write_text("corrupted")

    with pytest.raises(ValueError, match="different input|differs from the import record"):
        store.import_trajectory(str(source), filename, "identity")


@pytest.mark.asyncio
async def test_missing_original_requires_published_copy(runtime, tmp_path, monkeypatch):
    _, store, _ = runtime
    source = tmp_path / "original.log"
    source.write_text("evidence")

    def fail(*_args):
        raise RuntimeError("publication interrupted")

    monkeypatch.setattr("reme_procedure_memory.trajectory.os.link", fail)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        store.import_trajectory(str(source), "saved.log", "unpublished")
    source.unlink()
    with pytest.raises(FileNotFoundError):
        store.import_trajectory(str(source), "saved.log", "unpublished")
    with pytest.raises(FileNotFoundError):
        store.import_trajectory(str(source), "saved.log", "new-import")


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["../evil.txt", "/tmp/evil", "a/b.log", "a\\b.log", "..", "CON.txt", ""])
async def test_filename_containment(runtime, tmp_path, filename):
    _, store, _ = runtime
    source = tmp_path / "safe"
    source.write_text("safe")
    with pytest.raises(ValueError):
        store.import_trajectory(str(source), filename, "bad")


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,text", [("bad.json", "{"), ("bad.jsonl", "{}\nnot json"), ("binary.pdf", "opaque")])
async def test_invalid_format_retains_copy(runtime, tmp_path, filename, text):
    _, store, _ = runtime
    source = tmp_path / filename
    source.write_text(text)
    record = store.import_trajectory(str(source), filename, "bad-format")
    copied = store.source_path(record["source"])
    with pytest.raises(ValueError):
        await prepare_text(copied)
    assert copied.read_text() == text


@pytest.mark.asyncio
async def test_source_and_experience_symlink_escape(runtime, tmp_path):
    _, store, _ = runtime
    external = tmp_path / "outside.txt"
    external.write_text("private")
    (store.trajectory_root / "escape.txt").symlink_to(external)
    with pytest.raises(ValueError):
        store.source_path("session/trajectories/escape.txt")
    (store.procedure_root / "escape.json").symlink_to(external)
    with pytest.raises(ValueError):
        await store.read_experience("procedures/escape.json")
