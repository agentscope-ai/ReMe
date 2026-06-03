"""Tests for the WriteStep ``metadata`` param and the per-path write lock.

The ``metadata`` feature lets callers extend the on-disk frontmatter beyond
the two reserved fields (``name`` / ``description``) without touching the
step interface. Reserved keys inside the dict are ignored — explicit
top-level parameters always win.

The per-path lock serializes concurrent write_step invocations targeting
the same path within a single process. We exercise it by firing many
concurrent writes at one path and asserting the final state is consistent
(the lock guarantees the last write's bytes land intact, not a torn
interleaving).
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

import frontmatter

from reme4.components.file_store import LocalFileStore
from reme4.steps.file_io import write as crud_write

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class temp_chdir:
    """Context manager to temporarily chdir into a path and restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


def _run(coro):
    asyncio.run(coro)


async def _make_store() -> LocalFileStore:
    store = LocalFileStore(name="t_write_meta", embedding_store="")
    await store.start()
    return store


async def _write(store: LocalFileStore, **kwargs):
    step = crud_write.WriteStep(file_store=store)
    await step(**kwargs)
    return step.context.response


# -- metadata expansion ------------------------------------------------------


def test_write_metadata_extends_frontmatter():
    """``metadata={"tags": [...]}`` ends up as a frontmatter field on disk."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(
                store,
                path="note.md",
                name="Hello",
                description="A note",
                content="body text",
                metadata={"tags": ["alpha", "beta"], "priority": 3},
            )
            assert resp.success is True, resp
            on_disk = (Path(tmp) / "note.md").read_text(encoding="utf-8")
            post = frontmatter.loads(on_disk)
            assert post.metadata["name"] == "Hello"
            assert post.metadata["description"] == "A note"
            assert post.metadata["tags"] == ["alpha", "beta"]
            assert post.metadata["priority"] == 3
            assert post.content.strip() == "body text"
            await store.close()
        print("✓ test_write_metadata_extends_frontmatter passed")

    _run(run())


def test_write_metadata_reserved_keys_ignored():
    """``name`` / ``description`` inside ``metadata`` are dropped; explicit args win."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(
                store,
                path="note.md",
                name="ExplicitName",
                description="ExplicitDesc",
                content="body",
                metadata={"name": "DroppedName", "description": "DroppedDesc", "tag": "kept"},
            )
            assert resp.success is True, resp
            post = frontmatter.loads((Path(tmp) / "note.md").read_text(encoding="utf-8"))
            assert post.metadata["name"] == "ExplicitName"
            assert post.metadata["description"] == "ExplicitDesc"
            assert post.metadata["tag"] == "kept"
            await store.close()
        print("✓ test_write_metadata_reserved_keys_ignored passed")

    _run(run())


def test_write_no_metadata_preserves_legacy_shape():
    """No ``metadata`` arg → frontmatter still contains only name/description."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(
                store,
                path="note.md",
                name="N",
                description="D",
                content="body",
            )
            assert resp.success is True, resp
            post = frontmatter.loads((Path(tmp) / "note.md").read_text(encoding="utf-8"))
            assert set(post.metadata.keys()) == {"name", "description"}
            await store.close()
        print("✓ test_write_no_metadata_preserves_legacy_shape passed")

    _run(run())


def test_write_non_md_drops_metadata():
    """Non-markdown target: metadata silently dropped, body written verbatim."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(
                store,
                path="note.txt",
                name="N",
                description="D",
                content="plain body",
                metadata={"tag": "ignored"},
            )
            assert resp.success is True, resp
            on_disk = (Path(tmp) / "note.txt").read_text(encoding="utf-8")
            assert on_disk == "plain body"
            await store.close()
        print("✓ test_write_non_md_drops_metadata passed")

    _run(run())


# -- per-path lock -----------------------------------------------------------


def test_write_lock_serializes_concurrent_writes():
    """Many concurrent writes at one path land cleanly — no torn frontmatter.

    Without the lock, concurrent writers can interleave reads-of-existence
    and writes-of-bytes; with it, each write either runs before or after
    every other write. The on-disk file at the end must parse as valid
    frontmatter with name matching exactly one of the writers.
    """

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            n = 16
            await asyncio.gather(
                *(
                    _write(
                        store,
                        path="shared.md",
                        name=f"writer-{i}",
                        description=f"desc-{i}",
                        content=f"body-{i}",
                    )
                    for i in range(n)
                ),
            )
            on_disk = (Path(tmp) / "shared.md").read_text(encoding="utf-8")
            post = frontmatter.loads(on_disk)
            assert post.metadata.get("name", "").startswith("writer-"), post.metadata
            assert post.metadata.get("description", "").startswith("desc-"), post.metadata
            assert post.content.strip().startswith("body-"), post.content
            # The body, name, and description must all come from the SAME write
            # (no interleaving). Extract the index from each.
            idx_name = post.metadata["name"].split("-", 1)[1]
            idx_desc = post.metadata["description"].split("-", 1)[1]
            idx_body = post.content.strip().split("-", 1)[1]
            assert idx_name == idx_desc == idx_body, (idx_name, idx_desc, idx_body)
            await store.close()
        print("✓ test_write_lock_serializes_concurrent_writes passed")

    _run(run())
