"""Keep ranged reads aligned with universal-newline chunking and large-file reads."""

import asyncio
import time

import pytest

from reme.components import ApplicationContext
from reme.components.file_chunker import DefaultFileChunker
from reme.steps.file_io import edit as edit_module
from reme.steps.file_io import read as read_module


@pytest.mark.parametrize("newline,suffix", [("\n", ".md"), ("\r\n", ".txt")])
@pytest.mark.parametrize("found", [False, True])
def test_edit_repeated_prefix_search(tmp_path, newline, suffix, found):
    """A small stale edit must not spend seconds retrying a long shared prefix."""
    path = tmp_path / ("note" + suffix)
    source = ("x" + newline) * 100_000
    if found:
        source += "z"
    path.write_bytes(source.encode("utf-8"))
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        editor = edit_module.EditStep(app_context=app_context)
        started = time.process_time()
        await editor(path=path.name, old="x\n" * 2_000 + "z", new="replacement")
        cpu_seconds = time.process_time() - started
        assert editor.context.response.success is found
        if found:
            expected = ("x" + newline) * 98_000 + "replacement" + ("\n" if suffix == ".md" else "")
            assert "Replaced 1 occurrence(s)" in editor.context.response.answer
            assert path.read_bytes() == expected.encode("utf-8")
        else:
            assert "not found" in editor.context.response.answer
            assert path.read_bytes() == source.encode("utf-8")
        # CPU time excludes delays from other processes on shared CI runners.
        # The literal-search implementation is orders of magnitude below this bound.
        assert cpu_seconds < 1.0, f"repeated-prefix search used {cpu_seconds:.3f}s CPU"

    asyncio.run(run())


@pytest.mark.parametrize(
    "source,old,expected",
    [
        ("\r\na\r\nb\r\n", "a\nb", "\r\nR\r\n"),
        ("\r\na\r\nb", "\na", "R\r\nb"),
        ("a\r\nb\r\n", "b\n", "a\r\nR"),
        ("\r\n\r\n", "\n", "RR"),
        ("a\r\nb\rc\nd", "\n", "aRbRcRd"),
        ("a\r\n\r\nb", "\n\n", "aRb"),
        ("a\r\na\na\ra", "a\na", "R\nR"),
        ("\r\na\u2028b\x0bc\r\n", "a\u2028b\x0bc\n", "\r\nR"),
    ],
)
def test_edit_normalized_match_boundaries(tmp_path, source, old, expected):
    """Mapped spans may start/end at a newline and must remain non-overlapping."""
    path = tmp_path / "note.txt"
    path.write_bytes(source.encode("utf-8"))
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        editor = edit_module.EditStep(app_context=app_context)
        await editor(path=path.name, old=old, new="R")
        assert editor.context.response.success is True
        assert path.read_bytes() == expected.encode("utf-8")

    asyncio.run(run())


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
@pytest.mark.parametrize("trailing_newline", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r", "mixed"])
def test_read_line_ranges_match_chunker_and_streaming_path(tmp_path, monkeypatch, encoding, trailing_newline, newline):
    """The same bytes must produce the same line ranges on both read paths."""
    logical_lines = ["first\u2028inline", "second 中文", "third\x0binline", "fourth"]
    separators = ["\r\n", "\r", "\n"] if newline == "mixed" else [newline] * 3
    text = logical_lines[0] + "".join(separator + line for separator, line in zip(separators, logical_lines[1:]))
    if trailing_newline:
        text += "\r" if newline == "mixed" else newline
    path = tmp_path / "note.md"
    source = text.encode(encoding)
    path.write_bytes(source)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        # UTF-8 is the normal indexer configuration. UTF-16 still exercises the
        # read helper's BOM detection without asserting byte-chunker geometry.
        if encoding == "utf-8":
            _, chunks = await DefaultFileChunker(encoding=encoding, app_context=app_context).chunk(path)
            assert chunks[0].start_line == 1
            assert chunks[0].end_line == len(logical_lines)

        for threshold in (len(source) + 1, 1):
            monkeypatch.setattr(read_module, "MAX_FILE_READ_BYTES", threshold)
            step = read_module.ReadStep(app_context=app_context)
            await step(path="note.md", start_line=2, end_line=3)
            assert step.context.response.success is True
            assert step.context.response.answer == "\n".join(logical_lines[1:3])

            complete = read_module.ReadStep(app_context=app_context)
            await complete(path="note.md")
            assert complete.context.response.success is True
            assert complete.context.response.answer == "\n".join(logical_lines)

            past_end = read_module.ReadStep(app_context=app_context)
            await past_end(path="note.md", start_line=5)
            assert past_end.context.response.success is False
            assert "exceeds file length (4 lines)" in past_end.context.response.answer

        assert path.read_bytes() == source

    asyncio.run(run())


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
@pytest.mark.parametrize("trailing_newline", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_read_then_edit_multiline(tmp_path, encoding, trailing_newline, newline, suffix):
    """Read output remains usable as an edit needle without changing write rules."""
    text = f"first\u2028inline{newline}second\x0binline"
    if trailing_newline:
        text += newline
    path = tmp_path / ("note" + suffix)
    source = text.encode(encoding)
    path.write_bytes(source)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        reader = read_module.ReadStep(app_context=app_context)
        await reader(path=path.name)
        assert reader.context.response.success is True
        assert reader.context.response.answer == "first\u2028inline\nsecond\x0binline"
        assert path.read_bytes() == source

        editor = edit_module.EditStep(app_context=app_context)
        await editor(path=path.name, old=reader.context.response.answer, new="replacement")
        assert editor.context.response.success is True
        assert "Replaced 1 occurrence(s)" in editor.context.response.answer
        ending = "\n" if suffix == ".md" else newline if trailing_newline else ""
        assert path.read_bytes() == ("replacement" + ending).encode(encoding)

    asyncio.run(run())


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
@pytest.mark.parametrize("filename,encoding", [("note.txt", "utf-8"), ("note.txt", "utf-16"), ("note.csv", "gbk")])
def test_ranged_read_then_edit_preserves_unmatched_bytes(tmp_path, newline, filename, encoding):
    """Only the matched span changes; encoding, other newlines and replacement literals survive."""
    prefix = "开头" + newline
    target = "first [1].*" + newline + r"second (2)\path"
    suffix = newline + "结尾" + newline
    path = tmp_path / filename
    path.write_bytes((prefix + target + suffix).encode(encoding))
    replacement = r"\1 replacement" + "\n" + r"next\g<0>"
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        reader = read_module.ReadStep(app_context=app_context)
        await reader(path=filename, start_line=2, end_line=3)
        assert reader.context.response.success is True
        editor = edit_module.EditStep(app_context=app_context)
        await editor(path=filename, old=reader.context.response.answer, new=replacement)
        assert editor.context.response.success is True
        assert path.read_bytes() == (prefix + replacement + suffix).encode(encoding)

    asyncio.run(run())


def test_edit_normalized_needle_replaces_all_newline_styles(tmp_path):
    """A mixed-newline body must not leave non-LF matches behind."""
    path = tmp_path / "note.txt"
    path.write_bytes(b"prefix\r\nfirst\r\nsecond\nmiddle\rfirst\nsecond\r\nfirst\rsecond\r\nsuffix")
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        editor = edit_module.EditStep(app_context=app_context)
        await editor(path=path.name, old="first\nsecond", new="done")
        assert editor.context.response.success is True
        assert "Replaced 3 occurrence(s)" in editor.context.response.answer
        assert path.read_bytes() == b"prefix\r\ndone\nmiddle\rdone\r\ndone\r\nsuffix"

    asyncio.run(run())


@pytest.mark.parametrize("separator", ["\r\n", "\r\n\r\n", "\r\r", "\n\n"])
def test_edit_does_not_treat_crlf_as_two_line_breaks(tmp_path, separator):
    """A normalized blank line cannot match a single CRLF separator."""
    path = tmp_path / "note.txt"
    source = ("first" + separator + "second").encode("utf-8")
    path.write_bytes(source)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        editor = edit_module.EditStep(app_context=app_context)
        await editor(path=path.name, old="first\n\nsecond", new="wrong")
        if separator == "\r\n":
            assert editor.context.response.success is False
            assert "not found" in editor.context.response.answer
            assert path.read_bytes() == source
        else:
            assert editor.context.response.success is True
            assert path.read_bytes() == b"wrong"

    asyncio.run(run())


def test_edit_normalized_needle_preserves_frontmatter(tmp_path):
    """Universal-newline matching stays inside the Markdown body."""
    path = tmp_path / "note.md"
    path.write_bytes(b"---\nname: first\ndescription: second\n---\nfirst\rsecond\rsuffix")
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        editor = edit_module.EditStep(app_context=app_context)
        await editor(path=path.name, old="first\nsecond", new="done")
        assert editor.context.response.success is True
        assert path.read_bytes() == b"---\ndescription: second\nname: first\n---\n\ndone\rsuffix\n"

        before = path.read_bytes()
        await editor(path=path.name, old="description: second\nname: first", new="wrong")
        assert editor.context.response.success is False
        assert path.read_bytes() == before

    asyncio.run(run())


@pytest.mark.parametrize("old", ["first", "first\rsecond", "first\r\nsecond"])
def test_edit_literal_needles_preserve_existing_matching(tmp_path, old):
    """Single-line and explicitly raw-newline needles retain exact replacement behavior."""
    path = tmp_path / "note.txt"
    source = "prefix\r\nfirst\rsecond / first\r\nsecond / first\nsecond\r\nsuffix"
    path.write_bytes(source.encode("utf-8"))
    app_context = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        editor = edit_module.EditStep(app_context=app_context)
        await editor(path=path.name, old=old, new="done")
        assert editor.context.response.success is True
        assert path.read_bytes() == source.replace(old, "done").encode("utf-8")

    asyncio.run(run())
