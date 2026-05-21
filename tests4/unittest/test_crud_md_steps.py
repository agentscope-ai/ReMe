"""End-to-end tests for reme4 crud_md steps: spawn `reme4 start`, drive via HTTP,
verify responses, then shut down. Each test uses an isolated cwd so the working_dir
(.reme by default) does not collide.

CLI rule: `path=` is relative-only, rooted at the reme working_dir. A bare path with
no suffix auto-appends `.md`; non-`.md` suffix is rejected. Absolute paths are
rejected.
"""

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme4.utils import call_action, call_and_check, mock_reme_server

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class _temp_chdir:
    """chdir to path for the duration of the block; restore on exit."""

    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


def _run(coro):
    """Run an async coroutine on a fresh isolated event loop."""
    asyncio.run(coro)


def _seed_md(working_dir: Path, rel: str, body: str) -> Path:
    target = working_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Individual job tests
# ---------------------------------------------------------------------------


def test_read_relative_path():
    """`reme4 read path=Templates/Recipe.md` returns the file body from .reme/."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            body = "# Recipe\n\nMix flour and water.\n"
            _seed_md(working, "Templates/Recipe.md", body)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Templates/Recipe.md",
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "# Recipe" in str(r.get("answer", ""))
                        and "flour and water" in str(r.get("answer", ""))
                    ),
                )
        print("✓ test_read_relative_path passed")

    _run(run())


def test_read_no_suffix_autoappends_md():
    """A bare path with no suffix auto-appends `.md`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Templates/Recipe.md", "auto-md\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Templates/Recipe",
                    validator=lambda r: (
                        isinstance(r, dict) and r.get("success") is True and "auto-md" in str(r.get("answer", ""))
                    ),
                )
        print("✓ test_read_no_suffix_autoappends_md passed")

    _run(run())


def test_read_line_range():
    """start_line / end_line slice the file 1-based, inclusive."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Notes.md", "L1\nL2\nL3\nL4\nL5\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Notes.md",
                    start_line=2,
                    end_line=4,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "L2" in str(r["answer"])
                        and "L3" in str(r["answer"])
                        and "L4" in str(r["answer"])
                        and "L1" not in str(r["answer"])
                        and "L5" not in str(r["answer"])
                    ),
                )
        print("✓ test_read_line_range passed")

    _run(run())


def test_read_absolute_path_accepted():
    """Absolute paths are accepted (a log warning is emitted but the read proceeds)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            target = _seed_md(working, "Abs.md", "x\n")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path=str(target.resolve()),
                )
                if not (
                    isinstance(result, dict) and result.get("success") is True and "x" in str(result.get("answer", ""))
                ):
                    raise AssertionError(f"expected absolute-path read to succeed, got {result!r}")
        print("✓ test_read_absolute_path_accepted passed")

    _run(run())


def test_read_non_md_degraded():
    """Paths whose suffix is not `.md` are read in compatibility mode."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "data/foo.txt", "plain-text body\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="data/foo.txt",
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "plain-text body" in str(r.get("answer", ""))
                    ),
                )
        print("✓ test_read_non_md_degraded passed")

    _run(run())


def test_read_missing_file():
    """Reading a non-existent file should fail with a clear error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="NotThere.md",
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "does not exist" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected missing-file rejection, got {result!r}")
        print("✓ test_read_missing_file passed")

    _run(run())


def test_read_start_after_end():
    """start_line > end_line is invalid."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Range.md", "a\nb\nc\n")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="Range.md",
                    start_line=3,
                    end_line=1,
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "start_line" in str(result.get("answer", ""))
                ):
                    raise AssertionError(f"expected start>end rejection, got {result!r}")
        print("✓ test_read_start_after_end passed")

    _run(run())


def test_read_start_line_exceeds_total():
    """start_line beyond total line count is invalid."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Short.md", "only-one-line\n")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="Short.md",
                    start_line=99,
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "exceeds" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected exceeds-length rejection, got {result!r}")
        print("✓ test_read_start_line_exceeds_total passed")

    _run(run())


def test_read_truncation():
    """A file larger than DEFAULT_MAX_BYTES triggers truncation with a continuation notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            # Seed > DEFAULT_MAX_BYTES (50 KiB) so the default truncation kicks in.
            body = "\n".join(f"line {i}" for i in range(8000)) + "\n"
            _seed_md(working, "Big.md", body)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Big.md",
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "truncated" in str(r["answer"])
                        and "start_line=" in str(r["answer"])
                    ),
                )
        print("✓ test_read_truncation passed")

    _run(run())


def test_read_empty_path_rejected():
    """An empty `path` should be rejected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action("read", host=host, port=port, path="")
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "required" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected `path` required rejection, got {result!r}")
        print("✓ test_read_empty_path_rejected passed")

    _run(run())


# ---------------------------------------------------------------------------
# write / edit / append tests
# ---------------------------------------------------------------------------


def test_write_basic_with_frontmatter():
    """`reme4 write path=... name=... description=... content=...` writes a YAML front matter block."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="Notes/A.md",
                    name="Greetings",
                    description="a friendly hello note",
                    content="# Hello",
                    validator=lambda r: (
                        isinstance(r, dict) and r.get("success") is True and "Wrote" in str(r.get("answer", ""))
                    ),
                )
            on_disk = (working / "Notes/A.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "name: Greetings" in on_disk
            assert "description: a friendly hello note" in on_disk
            assert "# Hello" in on_disk
        print("✓ test_write_basic_with_frontmatter passed")

    _run(run())


def test_write_no_suffix_autoappends_md():
    """`path` with no suffix gets `.md` appended."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="Notes/My",
                    content="x",
                    validator=lambda r: r.get("success") is True,
                )
            assert (working / "Notes/My.md").exists()
        print("✓ test_write_no_suffix_autoappends_md passed")

    _run(run())


def test_write_overwrites_with_notice():
    """Writing into an existing path overwrites the file and surfaces a system notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Existing.md", "old\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="Existing.md",
                    content="new",
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "Wrote" in str(r.get("answer", ""))
                        and "already existed" in str(r.get("answer", ""))
                        and "overwritten" in str(r.get("answer", ""))
                    ),
                )
            # File body has been replaced.
            on_disk = (working / "Existing.md").read_text(encoding="utf-8")
            assert "new" in on_disk and "old" not in on_disk, on_disk
        print("✓ test_write_overwrites_with_notice passed")

    _run(run())


def test_write_creates_parent_dirs():
    """Nested-non-existent parents are auto-created."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="a/b/c/D.md",
                    content="hi",
                    validator=lambda r: r.get("success") is True,
                )
            assert (working / "a/b/c/D.md").exists()
        print("✓ test_write_creates_parent_dirs passed")

    _run(run())


def test_write_no_frontmatter_when_all_empty():
    """When both `name` and `description` are empty strings, the file is body-only.

    The CLI schema declares them required, but the step is intentionally lenient
    so manual calls without these fields don't fail catastrophically."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="Plain.md",
                    name="",
                    description="",
                    content="# Hello",
                    validator=lambda r: r.get("success") is True,
                )
            on_disk = (working / "Plain.md").read_text(encoding="utf-8")
            assert not on_disk.startswith("---"), on_disk
            assert "# Hello" in on_disk
        print("✓ test_write_no_frontmatter_when_all_empty passed")

    _run(run())


def test_write_ignores_arbitrary_extra_fields():
    """Extra kwargs beyond name/description are silently ignored (schema is strict)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="Custom.md",
                    name="My Note",
                    description="short summary",
                    content="body",
                    # Extras below should NOT appear in front matter under the
                    # hardcoded-fields schema.
                    title="ignored",
                    author="ignored",
                    tags='["x","y"]',
                    validator=lambda r: r.get("success") is True,
                )
            on_disk = (working / "Custom.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "name: My Note" in on_disk
            assert "description: short summary" in on_disk
            assert "title:" not in on_disk
            assert "author:" not in on_disk
            assert "tags:" not in on_disk
            assert "body" in on_disk
        print("✓ test_write_ignores_arbitrary_extra_fields passed")

    _run(run())


def test_write_only_description_present():
    """Step is lenient: providing only `description` works; missing `name` is skipped."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="OnlyDesc.md",
                    description="just a description",
                    content="body",
                    validator=lambda r: r.get("success") is True,
                )
            on_disk = (working / "OnlyDesc.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "description: just a description" in on_disk
            assert "name:" not in on_disk
        print("✓ test_write_only_description_present passed")

    _run(run())


def test_edit_global_replace():
    """`reme4 edit` replaces every occurrence of `old` with `new`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "E.md", "foo bar foo\nfoo\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "edit",
                    host=host,
                    port=port,
                    path="E.md",
                    old="foo",
                    new="qux",
                    validator=lambda r: (
                        r.get("success") is True and "3" in str(r.get("answer", ""))  # 3 replacements
                    ),
                )
            assert (working / "E.md").read_text(encoding="utf-8") == "qux bar qux\nqux\n"
        print("✓ test_edit_global_replace passed")

    _run(run())


def test_edit_old_not_found():
    """`old` absent in the file → success=False."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "E.md", "hello world\n")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "edit",
                    host=host,
                    port=port,
                    path="E.md",
                    old="absent",
                    new="x",
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "not found" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected not-found rejection, got {result!r}")
            # File unchanged.
            assert (working / "E.md").read_text(encoding="utf-8") == "hello world\n"
        print("✓ test_edit_old_not_found passed")

    _run(run())


def test_edit_missing_file():
    """Editing a non-existent file should fail."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "edit",
                    host=host,
                    port=port,
                    path="NotThere.md",
                    old="x",
                    new="y",
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "does not exist" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected missing-file rejection, got {result!r}")
        print("✓ test_edit_missing_file passed")

    _run(run())


def test_edit_skips_frontmatter():
    """A match present in both front matter and body is replaced only in the body."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            body = (
                "---\n"
                "name: alpha\n"
                "description: alpha-doc\n"
                "---\n"
                "intro paragraph mentioning alpha and alpha again.\n"
            )
            _seed_md(working, "WithFM.md", body)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "edit",
                    host=host,
                    port=port,
                    path="WithFM.md",
                    old="alpha",
                    new="beta",
                    validator=lambda r: (
                        r.get("success") is True and "2" in str(r.get("answer", ""))  # 2 body occurrences only
                    ),
                )
            on_disk = (working / "WithFM.md").read_text(encoding="utf-8")
            # Front matter untouched.
            assert "name: alpha" in on_disk, on_disk
            assert "description: alpha-doc" in on_disk, on_disk
            # Body fully rewritten.
            assert "beta and beta" in on_disk, on_disk
            assert "alpha and alpha" not in on_disk, on_disk
        print("✓ test_edit_skips_frontmatter passed")

    _run(run())


def test_edit_match_only_in_frontmatter_fails():
    """If `old` appears ONLY inside front matter, edit reports not-found and writes nothing."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            body = "---\nname: secret\ndescription: nope\n---\nplain body without the keyword.\n"
            _seed_md(working, "FMOnly.md", body)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "edit",
                    host=host,
                    port=port,
                    path="FMOnly.md",
                    old="secret",
                    new="leaked",
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "not found" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected not-found rejection, got {result!r}")
            # File untouched.
            assert (working / "FMOnly.md").read_text(encoding="utf-8") == body
        print("✓ test_edit_match_only_in_frontmatter_fails passed")

    _run(run())


def test_append_basic():
    """Append adds content to the end of an existing file."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "A.md", "L1\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "append",
                    host=host,
                    port=port,
                    path="A.md",
                    content="L2\n",
                    validator=lambda r: r.get("success") is True and "Appended" in r["answer"],
                )
            assert (working / "A.md").read_text(encoding="utf-8") == "L1\nL2\n"
        print("✓ test_append_basic passed")

    _run(run())


def test_append_concatenates_verbatim():
    """Append concatenates content verbatim — no implicit newline insertion."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "A.md", "abc")  # no trailing newline
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "append",
                    host=host,
                    port=port,
                    path="A.md",
                    content="def",
                    validator=lambda r: r.get("success") is True,
                )
            assert (working / "A.md").read_text(encoding="utf-8") == "abcdef"
        print("✓ test_append_concatenates_verbatim passed")

    _run(run())


def test_append_auto_creates_missing_file():
    """Append on a non-existent path creates the file and surfaces a system notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "append",
                    host=host,
                    port=port,
                    path="Fresh.md",
                    content="hello\n",
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "Appended" in str(r.get("answer", ""))
                        and "auto-created" in str(r.get("answer", ""))
                    ),
                )
            assert (working / "Fresh.md").read_text(encoding="utf-8") == "hello\n"
        print("✓ test_append_auto_creates_missing_file passed")

    _run(run())


def test_append_empty_content_on_existing_file_is_noop():
    """Appending empty content to an existing file leaves it unchanged."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "A.md", "L1\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "append",
                    host=host,
                    port=port,
                    path="A.md",
                    content="",
                    validator=lambda r: r.get("success") is True and "0 bytes" in str(r.get("answer", "")),
                )
            assert (working / "A.md").read_text(encoding="utf-8") == "L1\n"
        print("✓ test_append_empty_content_on_existing_file_is_noop passed")

    _run(run())


def test_append_empty_content_creates_empty_file():
    """Appending empty content to a missing path creates an empty file (with notice)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "append",
                    host=host,
                    port=port,
                    path="Empty.md",
                    content="",
                    validator=lambda r: (r.get("success") is True and "auto-created" in str(r.get("answer", ""))),
                )
            target = working / "Empty.md"
            assert target.exists() and target.read_text(encoding="utf-8") == ""
        print("✓ test_append_empty_content_creates_empty_file passed")

    _run(run())


# ---------------------------------------------------------------------------
# Non-markdown degraded-mode tests
# ---------------------------------------------------------------------------


def test_write_non_md_skips_frontmatter():
    """Writing to a non-md path skips name/description and emits a recommendation notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "write",
                    host=host,
                    port=port,
                    path="data/notes.txt",
                    name="Greetings",
                    description="should be ignored",
                    content="# Hello",
                    validator=lambda r: (
                        r.get("success") is True
                        and "Wrote" in str(r.get("answer", ""))
                        and "non-markdown" in str(r.get("answer", "")).lower()
                    ),
                )
            on_disk = (working / "data/notes.txt").read_text(encoding="utf-8")
            assert not on_disk.startswith("---"), on_disk
            assert "name: Greetings" not in on_disk
            assert on_disk == "# Hello"
        print("✓ test_write_non_md_skips_frontmatter passed")

    _run(run())


def test_edit_non_md_full_text():
    """Editing a non-md path operates on the full file body (no frontmatter parsing)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            # A YAML-looking header that would otherwise be stripped as frontmatter.
            body = "---\nname: keep-me\n---\nfoo bar foo\n"
            _seed_md(working, "data/code.txt", body)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "edit",
                    host=host,
                    port=port,
                    path="data/code.txt",
                    old="keep-me",
                    new="replaced",
                    validator=lambda r: (
                        r.get("success") is True
                        and "1" in str(r.get("answer", ""))
                        and "non-markdown" in str(r.get("answer", "")).lower()
                    ),
                )
            on_disk = (working / "data/code.txt").read_text(encoding="utf-8")
            assert "name: replaced" in on_disk, on_disk
            assert "foo bar foo" in on_disk, on_disk
        print("✓ test_edit_non_md_full_text passed")

    _run(run())


def test_append_non_md_warns():
    """Appending to a non-md file succeeds and surfaces the compatibility notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "data/log.txt", "line1\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "append",
                    host=host,
                    port=port,
                    path="data/log.txt",
                    content="line2\n",
                    validator=lambda r: (
                        r.get("success") is True and "non-markdown" in str(r.get("answer", "")).lower()
                    ),
                )
            assert (working / "data/log.txt").read_text(encoding="utf-8") == "line1\nline2\n"
        print("✓ test_append_non_md_warns passed")

    _run(run())


def test_read_non_utf8_encoding():
    """A GBK-encoded legacy file (e.g. CN-Windows CSV) is decoded via the GBK fallback."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            target = working / "data.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "姓名,职业\n你好世界,工程师\n"
            target.write_bytes(text.encode("gbk"))
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="data.csv",
                    validator=lambda r: (r.get("success") is True and "你好世界" in str(r.get("answer", ""))),
                )
        print("✓ test_read_non_utf8_encoding passed")

    _run(run())


def test_append_preserves_gbk_encoding():
    """Appending to a GBK file re-encodes new content in GBK (no UTF-8 corruption)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            target = working / "data.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            existing = ("姓名,年龄\n张三,30\n" * 20).encode("gbk")
            target.write_bytes(existing)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "append",
                    host=host,
                    port=port,
                    path="data.csv",
                    content="李四,25\n",
                    validator=lambda r: r.get("success") is True,
                )
            # File must round-trip as GBK; UTF-8 decoding would fail or yield mojibake.
            raw = target.read_bytes()
            assert raw.endswith("李四,25\n".encode("gbk")), raw[-20:]
            decoded = raw.decode("gbk")
            assert "张三" in decoded and "李四" in decoded
        print("✓ test_append_preserves_gbk_encoding passed")

    _run(run())


def test_edit_preserves_gbk_encoding():
    """Editing a GBK file keeps the file encoded in GBK after the rewrite."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            target = working / "notes.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "原始内容,占位\n" * 20
            target.write_bytes(text.encode("gbk"))
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "edit",
                    host=host,
                    port=port,
                    path="notes.csv",
                    old="原始内容",
                    new="替换后",
                    validator=lambda r: r.get("success") is True,
                )
            raw = target.read_bytes()
            # File still decodes as GBK (would raise if we'd silently converted to UTF-8).
            decoded = raw.decode("gbk")
            assert "替换后" in decoded and "原始内容" not in decoded
        print("✓ test_edit_preserves_gbk_encoding passed")

    _run(run())


def test_read_utf8_bom():
    """Reading a UTF-8 file with BOM strips the BOM transparently."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            target = working / "bom.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xef\xbb\xbfhello world\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="bom.txt",
                    validator=lambda r: (
                        r.get("success") is True
                        and "hello world" in str(r.get("answer", ""))
                        and "﻿" not in str(r.get("answer", ""))
                    ),
                )
        print("✓ test_read_utf8_bom passed")

    _run(run())


# ---------------------------------------------------------------------------
# Aggregate test: reuse one server instance for all read cases (faster).
# ---------------------------------------------------------------------------


def test_all_read_cases_one_server():
    """Run multiple read scenarios against a single shared server for efficiency."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Templates/Recipe.md", "# Recipe\nbody\n")
            _seed_md(working, "Notes.md", "L1\nL2\nL3\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Templates/Recipe.md",
                    validator=lambda r: r.get("success") is True and "# Recipe" in r["answer"],
                )
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Notes",
                    validator=lambda r: r.get("success") is True and "L1" in r["answer"],
                )
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Notes.md",
                    start_line=2,
                    end_line=2,
                    validator=lambda r: r.get("success") is True and r["answer"].strip() == "L2",
                )
        print("✓ test_all_read_cases_one_server passed")

    _run(run())


if __name__ == "__main__":
    print("\n=== reme4 crud_md (read) E2E tests ===")
    test_read_relative_path()
    test_read_no_suffix_autoappends_md()
    test_read_line_range()
    test_read_non_md_degraded()
    test_read_missing_file()
    test_read_start_after_end()
    test_read_start_line_exceeds_total()
    test_read_truncation()
    test_read_empty_path_rejected()
    test_all_read_cases_one_server()
    print("\n=== reme4 crud_md (write/edit/append) E2E tests ===")
    test_write_basic_with_frontmatter()
    test_write_no_suffix_autoappends_md()
    test_write_overwrites_with_notice()
    test_write_creates_parent_dirs()
    test_write_no_frontmatter_when_all_empty()
    test_write_ignores_arbitrary_extra_fields()
    test_write_only_description_present()
    test_edit_global_replace()
    test_edit_old_not_found()
    test_edit_missing_file()
    test_edit_skips_frontmatter()
    test_edit_match_only_in_frontmatter_fails()
    test_append_basic()
    test_append_concatenates_verbatim()
    test_append_auto_creates_missing_file()
    test_append_empty_content_on_existing_file_is_noop()
    test_append_empty_content_creates_empty_file()
    print("\n=== reme4 crud_md (non-md degraded mode) E2E tests ===")
    test_write_non_md_skips_frontmatter()
    test_edit_non_md_full_text()
    test_append_non_md_warns()
    test_read_non_utf8_encoding()
    test_append_preserves_gbk_encoding()
    test_edit_preserves_gbk_encoding()
    test_read_utf8_bom()
    print("\n所有测试通过!")
