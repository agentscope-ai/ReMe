"""Tests for the frontmatter-only CRUD steps (``frontmatter_read`` /
``frontmatter_update`` / ``frontmatter_delete``).

Covers the suffix-less path contract shared with ``read`` / ``write`` /
``edit``: a bare path with no suffix auto-appends ``.md``. All three
siblings must agree on path handling, disclose the substitution via
``resolved_path`` in the response metadata, and report the actually
probed path in not-found errors instead of the caller's raw input.
"""

# pylint: disable=protected-access

from contextlib import chdir as temp_chdir
import tempfile
from datetime import date
from pathlib import Path

import frontmatter
import pytest

from reme.components.file_store import LocalFileStore
from reme.steps.file_io.frontmatter_delete import FrontmatterDeleteStep
from reme.steps.file_io.frontmatter_read import FrontmatterReadStep
from reme.steps.file_io.frontmatter_update import FrontmatterUpdateStep


def _seed(workspace: Path, rel: str, body: str) -> Path:
    target = workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


async def _make_store() -> LocalFileStore:
    store = LocalFileStore(name="t_fm", embedding_store="")
    await store.start()
    return store


async def _run(step_cls, store: LocalFileStore, **kwargs):
    step = step_cls(file_store=store)
    await step(**kwargs)
    return step.context.response


NOTE = "notes/post.md"
BODY = "---\nname: n\ntags:\n- a\n---\nbody\n"


@pytest.mark.asyncio
async def test_read_no_suffix_autoappends_md():
    """frontmatter_read on a suffix-less path resolves to the ``.md`` file."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="notes/post")
        assert resp.success is True
        assert resp.metadata["frontmatter"] == {"name": "n", "tags": ["a"]}
        assert resp.metadata["path"] == "notes/post"
        assert resp.metadata["resolved_path"] == "notes/post.md"
        await store.close()


@pytest.mark.asyncio
async def test_read_honors_injected_path_scope():
    """An injected allowed-path scope denies frontmatter reads outside of it."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()

        allowed = await _run(FrontmatterReadStep, store, path=NOTE, _allowed_paths=[NOTE])
        assert allowed.success is True
        assert allowed.metadata["frontmatter"] == {"name": "n", "tags": ["a"]}

        denied = await _run(FrontmatterReadStep, store, path=NOTE, _allowed_paths=["notes/other.md"])
        assert denied.success is False
        assert denied.metadata["error"] == "no permission to access this file"
        await store.close()


@pytest.mark.asyncio
async def test_update_no_suffix_autoappends_md():
    """frontmatter_update on a suffix-less path resolves to the ``.md`` file."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        note = _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterUpdateStep, store, path="notes/post", metadata={"name": "renamed"})
        assert resp.success is True
        assert resp.metadata["updated"] == {"name": "renamed"}
        assert resp.metadata["resolved_path"] == "notes/post.md"
        assert "name: renamed" in note.read_text(encoding="utf-8")
        await store.close()


@pytest.mark.asyncio
async def test_update_honors_injected_frontmatter_key_scope():
    """Injected key scope rejects the entire update when any key is outside it."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        note = _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()

        before = note.read_bytes()
        resp = await _run(
            FrontmatterUpdateStep,
            store,
            path=NOTE,
            metadata={"tags": ["new"], "name": "renamed"},
            _allowed_frontmatter_keys=["tags"],
        )
        assert resp.success is False
        assert resp.metadata["error"] == "frontmatter key(s) not allowed: 'name'"
        assert note.read_bytes() == before

        resp = await _run(
            FrontmatterUpdateStep,
            store,
            path=NOTE,
            metadata={"tags": ["new"]},
            _allowed_frontmatter_keys=["tags"],
        )
        assert resp.success is True
        assert frontmatter.loads(note.read_text(encoding="utf-8")).metadata == {"name": "n", "tags": ["new"]}
        await store.close()


@pytest.mark.asyncio
async def test_update_empty_frontmatter_key_scope_denies_all_updates():
    """An empty injected list is restrictive, unlike an omitted constraint."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        note = _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        before = note.read_bytes()

        resp = await _run(
            FrontmatterUpdateStep,
            store,
            path=NOTE,
            metadata={"tags": ["new"]},
            _allowed_frontmatter_keys=[],
        )

        assert resp.success is False
        assert note.read_bytes() == before
        await store.close()


@pytest.mark.asyncio
async def test_delete_no_suffix_autoappends_md():
    """frontmatter_delete on a suffix-less path resolves to the ``.md`` file."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        note = _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterDeleteStep, store, path="notes/post", keys=["tags"])
        assert resp.success is True
        assert resp.metadata["deleted"] == ["tags"]
        assert resp.metadata["missing"] == []
        assert resp.metadata["resolved_path"] == "notes/post.md"
        assert "tags" not in note.read_text(encoding="utf-8")
        await store.close()


@pytest.mark.asyncio
async def test_family_consistent_on_suffixless_path():
    """read / update / delete all succeed on the same suffix-less path."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()

        resp = await _run(FrontmatterReadStep, store, path="notes/post")
        assert resp.success is True

        resp = await _run(FrontmatterUpdateStep, store, path="notes/post", metadata={"status": "done"})
        assert resp.success is True

        resp = await _run(FrontmatterDeleteStep, store, path="notes/post", keys=["status"])
        assert resp.success is True
        assert resp.metadata["deleted"] == ["status"]
        await store.close()


@pytest.mark.asyncio
async def test_read_missing_reports_probed_path():
    """not-found errors surface the probed ``.md`` path, not the raw input."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="digest/missing")
        assert resp.success is False
        assert resp.metadata["exists"] is False
        assert resp.metadata["path"] == "digest/missing"
        assert resp.metadata["resolved_path"] == "digest/missing.md"
        assert "digest/missing.md" in str(resp.answer)
        await store.close()


@pytest.mark.asyncio
async def test_update_missing_reports_probed_path():
    """frontmatter_update not-found errors mention the probed ``.md`` path."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _run(FrontmatterUpdateStep, store, path="digest/missing", metadata={"x": 1})
        assert resp.success is False
        assert resp.metadata["error"] == "digest/missing.md not found"
        assert resp.metadata["resolved_path"] == "digest/missing.md"
        assert "digest/missing.md" in str(resp.answer)
        await store.close()


@pytest.mark.asyncio
async def test_delete_missing_reports_probed_path():
    """frontmatter_delete not-found errors mention the probed ``.md`` path."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _run(FrontmatterDeleteStep, store, path="digest/missing", keys=["x"])
        assert resp.success is False
        assert resp.metadata["error"] == "digest/missing.md not found"
        assert resp.metadata["resolved_path"] == "digest/missing.md"
        assert "digest/missing.md" in str(resp.answer)
        await store.close()


@pytest.mark.asyncio
async def test_non_md_rejected_without_resolved_path():
    """Non-markdown targets are rejected by all three siblings; no substitution."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "notes/data.txt", "plain\n")
        store = await _make_store()

        for step_cls, kwargs in (
            (FrontmatterReadStep, {}),
            (FrontmatterUpdateStep, {"metadata": {"x": 1}}),
            (FrontmatterDeleteStep, {"keys": ["x"]}),
        ):
            resp = await _run(step_cls, store, path="notes/data.txt", **kwargs)
            assert resp.success is False
            assert "not markdown" in str(resp.answer).lower()
            assert "resolved_path" not in resp.metadata
        await store.close()


@pytest.mark.asyncio
async def test_explicit_md_path_has_no_resolved_path():
    """Explicit ``.md`` paths need no substitution, so no ``resolved_path``."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path=NOTE)
        assert resp.success is True
        assert "resolved_path" not in resp.metadata
        await store.close()


@pytest.mark.asyncio
async def test_read_answer_carries_the_frontmatter_itself():
    """The answer must contain the frontmatter, not just a key count.

    ``frontmatter_read`` is served straight to the LLM and MCP returns only
    ``response.answer``, so a count alone leaves the caller with nothing to
    act on. The suffix-less substitution is disclosed in the answer too.
    """
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="notes/post")
        assert resp.success is True
        assert resp.answer == 'Read frontmatter from notes/post.md (2 key(s)):\n- "name": "n"\n- "tags": ["a"]'
        assert resp.metadata["frontmatter"] == {"name": "n", "tags": ["a"]}
        await store.close()


@pytest.mark.asyncio
async def test_read_answer_preserves_value_structure_and_reports_empty():
    """Multi-line values stay on one line each; lists keep their shape; empty says so."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(
            Path(tmp),
            "notes/rich.md",
            "---\nname: Cobalt\ndescription: line one\n  line two\ntags:\n- a\n- b\nnested:\n  x: 1\n---\nbody\n",
        )
        _seed(Path(tmp), "notes/empty.md", "---\n---\nbody\n")
        store = await _make_store()

        resp = await _run(FrontmatterReadStep, store, path="notes/rich.md")
        assert resp.success is True
        assert resp.answer == (
            "Read frontmatter from notes/rich.md (4 key(s)):\n"
            '- "name": "Cobalt"\n'
            '- "description": "line one line two"\n'
            '- "tags": ["a", "b"]\n'
            '- "nested": {"x": 1}'
        )
        assert resp.metadata["frontmatter"] == {
            "name": "Cobalt",
            "description": "line one line two",
            "tags": ["a", "b"],
            "nested": {"x": 1},
        }

        resp = await _run(FrontmatterReadStep, store, path="notes/empty.md")
        assert resp.success is True
        assert resp.answer == "Read frontmatter from notes/empty.md (0 key(s)): (no keys)"
        assert resp.metadata["frontmatter"] == {}
        await store.close()


@pytest.mark.asyncio
async def test_read_answer_tolerates_rich_yaml_types():
    """Rendering must never turn a parseable frontmatter into a failed job.

    Regression test for the review findings: YAML dates (``datetime.date``)
    crashed ``json.dumps``, mixed mapping keys crashed ``sort_keys``, and
    comma-joined lists / collapsed whitespace lost value boundaries. All of
    these stay successful reads with the structured ``metadata`` untouched.
    """
    rich = """---
created: 2026-10-12
schedule:
  1: one
  label: two
tags: []
aliases:
  - "a, b"
  - c
command: "echo a  b"
description: |-
  line one
  line two
---
body
"""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "notes/types.md", rich)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="notes/types.md")
        assert resp.success is True
        expected_lines = [
            "Read frontmatter from notes/types.md (6 key(s)):",
            '- "created": "2026-10-12"',
            '- "schedule": {"1": "one", "label": "two"}',
            '- "tags": []',
            '- "aliases": ["a, b", "c"]',
            '- "command": "echo a  b"',
            '- "description": "line one\\nline two"',
        ]
        assert resp.answer == "\n".join(expected_lines)
        assert resp.metadata["frontmatter"] == {
            "created": date(2026, 10, 12),
            "schedule": {1: "one", "label": "two"},
            "tags": [],
            "aliases": ["a, b", "c"],
            "command": "echo a  b",
            "description": "line one\nline two",
        }
        await store.close()


@pytest.mark.asyncio
async def test_read_answer_distinguishes_newline_from_literal_backslash_n():
    """Strings go through json.dumps, so an actual newline and a
    two-character ``"\\n"`` sequence render as different escape
    sequences. Review follow-up: pin the answer so this distinction
    survives future refactors.
    """
    body = r"""---
real_newline: "line one\nline two"
literal_seq: "a\\nb"
---
body
"""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "notes/newline.md", body)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="notes/newline.md")
        assert resp.success is True
        assert '- "real_newline": "line one\\nline two"' in resp.answer
        # Literal two-character \n renders as a JSON-escaped backslash-n
        # (the JSON string contains a backslash + an n).
        assert '- "literal_seq": "a\\\\nb"' in resp.answer
        await store.close()


@pytest.mark.asyncio
async def test_read_answer_escapes_multiline_keys():
    """Keys go through the same JSON renderer as values, so a multiline
    YAML key stays a single ``- "key": value`` entry instead of breaking
    the one-key-per-line layout. Review finding: a key like
    ``first\\n- forged`` used to render as two lines and look like a
    second entry. The metadata keeps the original key.
    """
    body = r"""---
"first\n- forged": "value"
---
body
"""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "notes/keys.md", body)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="notes/keys.md")
        assert resp.success is True
        assert resp.answer == 'Read frontmatter from notes/keys.md (1 key(s)):\n- "first\\n- forged": "value"'
        assert resp.metadata["frontmatter"] == {"first\n- forged": "value"}
        await store.close()


@pytest.mark.asyncio
async def test_read_answer_mixed_mapping_key_types_render_via_json():
    """An int 1 and a string "1" stay distinct in YAML and in the
    metadata, but the JSON-rendered answer collapses both keys to the
    string "1": the fragment textually keeps both entries, while
    re-parsing it as JSON keeps only the latter value. Documented
    presentation limitation — the structured metadata is the source of
    truth for key types.
    """
    body = """---
mixed:
  1: "number"
  "1": "string"
---
body
"""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "notes/mixed_keys.md", body)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="notes/mixed_keys.md")
        assert resp.success is True
        # Both entries render textually; JSON's key coercion means a
        # re-parser sees only the last "1". The metadata keeps the types.
        assert '- "mixed": {"1": "number", "1": "string"}' in resp.answer
        assert resp.metadata["frontmatter"] == {
            "mixed": {1: "number", "1": "string"},
        }
        await store.close()
