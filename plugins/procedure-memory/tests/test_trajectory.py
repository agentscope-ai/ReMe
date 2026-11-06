"""Original-format imports, filename collisions and lossless segmentation."""

# pylint: disable=missing-function-docstring

import pytest

from reme_procedure_memory import trajectory
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


@pytest.mark.parametrize("line_count", [2500, 10000])
@pytest.mark.parametrize("chars_per_token", [1, 4])
def test_short_line_tokenization_work_is_linear(tmp_path, line_count, chars_per_token):
    counted = 0

    def count(text):
        nonlocal counted
        counted += len(text)
        return (len(text) + chars_per_token - 1) // chars_per_token

    text = ("x" * 78 + "\r\n") * line_count
    path = tmp_path / "short.log"
    path.write_bytes(text.encode())
    budget = TokenBudget(counter=count)
    chunks = list(segments(path, "utf-8", budget))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(0 < len(chunk["text"]) <= budget.limits["segment"] * chars_per_token for chunk in chunks)
    assert all(chunk["text"].endswith("\r\n") for chunk in chunks)
    assert [chunk["offset"] for chunk in chunks] == [
        sum(len(previous["text"]) for previous in chunks[:n]) for n in range(len(chunks))
    ]
    assert counted < len(text) * 1.6


def test_low_density_text_does_not_emit_the_calibration_sample(tmp_path):
    text = ("x" * 78 + "\r\n") * 2500
    path = tmp_path / "low-density.log"
    path.write_bytes(text.encode())
    budget = TokenBudget(counter=lambda value: (len(value) + 3) // 4)
    assert list(segments(path, "utf-8", budget)) == [{"offset": 0, "text": text}]


def test_final_partial_line_does_not_create_an_extra_segment(tmp_path):
    text = "x" * 100 + "\ntail"
    path = tmp_path / "final.log"
    path.write_bytes(text.encode())
    assert list(segments(path, "utf-8", TokenBudget(counter=len, segment=128))) == [{"offset": 0, "text": text}]


@pytest.mark.parametrize("limit", [64, 257])
@pytest.mark.parametrize("mode", ["bytes", "density_changes", "nonmonotonic", "zero"])
def test_segmentation_counter_work_is_bounded_even_when_predictions_fail(tmp_path, limit, mode):
    counted = 0

    def exact_count(value):
        if mode == "density_changes":
            return sum(32 if char == "!" else 1 for char in value)
        if mode == "nonmonotonic":
            return len(value) + (limit // 2 if value.endswith(("\r", "\n")) else 0)
        return 0 if mode == "zero" else len(value.encode())

    def count(value):
        nonlocal counted
        counted += len(value)
        return exact_count(value)

    text = ("a\n" * 256 + "!" * 1024 + "你好🙂\r\n" * 128) * 3
    path = tmp_path / "unpredictable.log"
    path.write_bytes(text.encode())
    chunks = list(segments(path, "utf-8", TokenBudget(counter=count, segment=limit)))
    assert counted < 8 * len(text)
    assert "".join(chunk["text"] for chunk in chunks) == text
    offset = 0
    for chunk in chunks:
        assert chunk["text"]
        assert exact_count(chunk["text"]) <= limit
        assert chunk["offset"] == offset
        offset += len(chunk["text"])


@pytest.mark.parametrize("window", [2, 7, 128])
def test_candidate_window_is_bounded_for_large_budgets(tmp_path, monkeypatch, window):
    monkeypatch.setattr(trajectory, "_MAX_SEGMENT_CHARS", window)
    counted = 0

    def count(value):
        nonlocal counted
        counted += len(value)
        assert len(value) <= window
        return len(value)

    text = "x" * 10000 + "\r\n" + "你好🙂\r\n" * 100
    path = tmp_path / "bounded-window.log"
    path.write_bytes(text.encode())
    chunks = list(segments(path, "utf-8", TokenBudget(counter=count, segment=100000)))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert counted == len(text)
    assert all(
        not (first["text"].endswith("\r") and second["text"].startswith("\n"))
        for first, second in zip(chunks, chunks[1:])
    )


def test_segment_budget_must_hold_one_character(tmp_path):
    path = tmp_path / "too-small.log"
    path.write_text("🙂", encoding="utf-8")
    budget = TokenBudget(counter=lambda value: len(value.encode()), segment=3)
    with pytest.raises(ValueError, match="cannot hold one character"):
        list(segments(path, "utf-8", budget))


@pytest.mark.parametrize("text", ["", "abc", "a\nb\nc\nd\n", "a\rb\rc\r", "a\r\nb\nc\rtail"])
def test_batched_segments_preserve_line_boundaries(tmp_path, text):
    path = tmp_path / "lines.log"
    path.write_bytes(text.encode())
    chunks = list(segments(path, "utf-8", TokenBudget(counter=len, segment=4)))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(0 < len(chunk["text"]) <= 4 for chunk in chunks)
    assert all(chunk["text"].endswith(("\r", "\n")) for chunk in chunks[:-1])


@pytest.mark.parametrize("segment_limit", [4, 17, 63])
def test_short_multibyte_lines_obey_exact_budget(tmp_path, segment_limit):
    text = "你好🙂\r\n" * 100 + "末尾"
    path = tmp_path / "short-unicode.log"
    path.write_bytes(text.encode())
    budget = TokenBudget(counter=lambda value: len(value.encode()), segment=segment_limit)
    chunks = list(segments(path, "utf-8", budget))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(0 < budget.count(chunk["text"]) <= segment_limit for chunk in chunks)


@pytest.mark.parametrize("text", ["a\n" * 200 + "你好🙂\r\n" * 200, "x" * 5000 + "\n" + "short\n" * 2000])
def test_batches_handle_changing_token_density_and_line_size(tmp_path, text):
    path = tmp_path / "mixed.log"
    path.write_bytes(text.encode())
    budget = TokenBudget(counter=lambda value: len(value.encode()), segment=127)
    chunks = list(segments(path, "utf-8", budget))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(0 < budget.count(chunk["text"]) <= 127 for chunk in chunks)
    assert [chunk["offset"] for chunk in chunks] == [
        sum(len(previous["text"]) for previous in chunks[:n]) for n in range(len(chunks))
    ]


def test_line_boundary_is_rechecked_with_nonmonotonic_counter(tmp_path):
    # A shorter prefix can cost more tokens when its ending changes.
    def count(text):
        return len(text) + (8 if text.endswith("\n") else 0)

    text = "abc\n" * 20 + "tail"
    path = tmp_path / "boundary.log"
    path.write_bytes(text.encode())
    budget = TokenBudget(counter=count, segment=10)
    chunks = list(segments(path, "utf-8", budget))
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(0 < budget.count(chunk["text"]) <= 10 for chunk in chunks)


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
