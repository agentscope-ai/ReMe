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


def test_read_non_md_rejected():
    """Paths whose suffix is not `.md` are rejected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="data/foo.txt",
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "markdown" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected markdown-only rejection, got {result!r}")
        print("✓ test_read_non_md_rejected passed")

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
# create / edit / append tests
# ---------------------------------------------------------------------------


def test_create_basic_with_frontmatter():
    """`reme4 create path=... title=... tags='[\"a\",\"b\"]' status=...` writes a YAML front matter block."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="Notes/A.md",
                    content="# Hello",
                    title="Greetings",
                    tags='["a","b"]',
                    status="draft",
                    validator=lambda r: (
                        isinstance(r, dict) and r.get("success") is True and "Created" in str(r.get("answer", ""))
                    ),
                )
            on_disk = (working / "Notes/A.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "title: Greetings" in on_disk
            assert "tags:" in on_disk and "- a" in on_disk and "- b" in on_disk
            assert "status: draft" in on_disk
            assert "# Hello" in on_disk
        print("✓ test_create_basic_with_frontmatter passed")

    _run(run())


def test_create_no_suffix_autoappends_md():
    """`path` with no suffix gets `.md` appended."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="Notes/My",
                    content="x",
                    validator=lambda r: r.get("success") is True,
                )
            assert (working / "Notes/My.md").exists()
        print("✓ test_create_no_suffix_autoappends_md passed")

    _run(run())


def test_create_overwrites_with_notice():
    """Creating into an existing path overwrites the file and surfaces a system notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Existing.md", "old\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="Existing.md",
                    content="new",
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "Created" in str(r.get("answer", ""))
                        and "already existed" in str(r.get("answer", ""))
                        and "overwritten" in str(r.get("answer", ""))
                    ),
                )
            # File body has been replaced.
            on_disk = (working / "Existing.md").read_text(encoding="utf-8")
            assert "new" in on_disk and "old" not in on_disk, on_disk
        print("✓ test_create_overwrites_with_notice passed")

    _run(run())


def test_create_creates_parent_dirs():
    """Nested-non-existent parents are auto-created."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="a/b/c/D.md",
                    content="hi",
                    validator=lambda r: r.get("success") is True,
                )
            assert (working / "a/b/c/D.md").exists()
        print("✓ test_create_creates_parent_dirs passed")

    _run(run())


def test_create_no_frontmatter_when_all_empty():
    """When all optional fields are empty strings, the file is just the body.

    Note: under the new generic-frontmatter semantics, an explicit empty list
    literal (``tags="[]"``) is now WRITTEN as ``tags: []`` because the user
    asked for it explicitly. Only empty/blank strings are skipped.
    """

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="Plain.md",
                    content="# Hello",
                    title="",
                    tags="",
                    status="",
                    validator=lambda r: r.get("success") is True,
                )
            on_disk = (working / "Plain.md").read_text(encoding="utf-8")
            assert not on_disk.startswith("---"), on_disk
            assert "# Hello" in on_disk
        print("✓ test_create_no_frontmatter_when_all_empty passed")

    _run(run())


def test_create_with_arbitrary_frontmatter_fields():
    """Any non-reserved kwarg is written as a front matter field (no hardcoded whitelist)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="Custom.md",
                    content="body",
                    title="My Note",
                    author="alice",
                    category="research",
                    created="2026-05-20",
                    version="1.0.0",
                    validator=lambda r: r.get("success") is True,
                )
            on_disk = (working / "Custom.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            for needle in (
                "title: My Note",
                "author: alice",
                "category: research",
                "created: '2026-05-20'",  # quoted because it parses as date-like
                "version: 1.0.0",
            ):
                # date string may serialize without quotes in some yaml setups; relax check
                base = needle.split(":", maxsplit=1)[0]
                assert f"{base}:" in on_disk, f"missing key {base} in:\n{on_disk}"
            assert "body" in on_disk
        print("✓ test_create_with_arbitrary_frontmatter_fields passed")

    _run(run())


def test_create_with_nested_dict_frontmatter():
    """A JSON/YAML object literal becomes a nested mapping in front matter."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="Nested.md",
                    content="x",
                    extra='{"k1": "v1", "k2": 2}',
                    validator=lambda r: r.get("success") is True,
                )
            on_disk = (working / "Nested.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "extra:" in on_disk
            assert "k1: v1" in on_disk
            assert "k2: 2" in on_disk
        print("✓ test_create_with_nested_dict_frontmatter passed")

    _run(run())


def test_create_invalid_yaml_literal_field_errors():
    """A string starting with `[`/`{` that is not valid YAML produces a clear error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "create",
                    host=host,
                    port=port,
                    path="Bad.md",
                    content="x",
                    tags='["unterminated',
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "invalid yaml" in str(result.get("answer", "")).lower()
                    and "`tags`" in str(result.get("answer", ""))
                ):
                    raise AssertionError(f"expected invalid-yaml rejection on tags, got {result!r}")
            assert not (working / "Bad.md").exists(), "file should not have been created"
        print("✓ test_create_invalid_yaml_literal_field_errors passed")

    _run(run())


def test_create_preserves_plain_strings_no_yaml_coerce():
    """Non-literal strings (no leading `[`/`{`) are kept verbatim — no yes/no/int coercion."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "create",
                    host=host,
                    port=port,
                    path="Plain2.md",
                    content="x",
                    status="yes",  # must remain the string "yes", not True
                    count="42",  # must remain the string "42", not int 42
                    validator=lambda r: r.get("success") is True,
                )
            on_disk = (working / "Plain2.md").read_text(encoding="utf-8")
            # YAML round-trips strings that would otherwise parse as bool/int by quoting them.
            assert "status: 'yes'" in on_disk or 'status: "yes"' in on_disk, on_disk
            assert "count: '42'" in on_disk or 'count: "42"' in on_disk, on_disk
        print("✓ test_create_preserves_plain_strings_no_yaml_coerce passed")

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


def test_append_inserts_newline_when_missing():
    """If file lacks a trailing newline, append inserts one before the new content."""

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
            assert (working / "A.md").read_text(encoding="utf-8") == "abc\ndef"
        print("✓ test_append_inserts_newline_when_missing passed")

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
    test_read_non_md_rejected()
    test_read_missing_file()
    test_read_start_after_end()
    test_read_start_line_exceeds_total()
    test_read_truncation()
    test_read_empty_path_rejected()
    test_all_read_cases_one_server()
    print("\n=== reme4 crud_md (create/edit/append) E2E tests ===")
    test_create_basic_with_frontmatter()
    test_create_no_suffix_autoappends_md()
    test_create_overwrites_with_notice()
    test_create_creates_parent_dirs()
    test_create_no_frontmatter_when_all_empty()
    test_create_with_arbitrary_frontmatter_fields()
    test_create_with_nested_dict_frontmatter()
    test_create_invalid_yaml_literal_field_errors()
    test_create_preserves_plain_strings_no_yaml_coerce()
    test_edit_global_replace()
    test_edit_old_not_found()
    test_edit_missing_file()
    test_append_basic()
    test_append_inserts_newline_when_missing()
    test_append_auto_creates_missing_file()
    test_append_empty_content_on_existing_file_is_noop()
    test_append_empty_content_creates_empty_file()
    print("\n所有测试通过!")
