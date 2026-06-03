"""Tests for the ``read_image`` step.

Style mirrors ``test_crud_steps.py`` — direct step invocation against a
freshly built ``LocalFileStore`` with the step's ``vault_path`` rooted at
``cwd()`` via chdir.

Coverage:
    - happy paths for known suffixes (png/jpeg)
    - oversized branch (``answer`` is a notice, ``metadata.oversized=True``)
    - unknown / missing suffix (compatibility mode; ``non_image_warning=True``)
    - error branches: missing file, directory, empty path, bad ``max_bytes``
"""

# pylint: disable=protected-access

import asyncio
import base64
import os
import tempfile
import warnings
from pathlib import Path

from reme4.components.file_store import LocalFileStore
from reme4.steps.file_io import read_image as crud_read_image

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
    store = LocalFileStore(name="t_img", embedding_store="")
    await store.start()
    return store


def _seed_bytes(rel: str, data: bytes) -> Path:
    """Drop raw bytes at ``cwd/rel``. Step is byte-level — no real PNG needed."""
    target = Path.cwd() / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


async def _read_image(store: LocalFileStore, *, step_kwargs: dict | None = None, **call_kwargs):
    """Run ReadImageStep; ``step_kwargs`` go to step init (kwargs/attrs)."""
    step = crud_read_image.ReadImageStep(file_store=store, **(step_kwargs or {}))
    await step(**call_kwargs)
    return step.context.response


def test_read_image_png():
    """``read_image path=img/cat.png`` returns base64 + ``image/png`` mime."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            payload = b"\x89PNG\r\n\x1a\n" + b"fake-png-body-bytes"
            _seed_bytes("img/cat.png", payload)
            store = await _make_store()
            resp = await _read_image(store, path="img/cat.png")
            assert resp.success is True, resp
            assert base64.b64decode(resp.answer) == payload, "base64 round-trip mismatch"
            assert resp.metadata["mime"] == "image/png", resp.metadata
            assert resp.metadata["size_bytes"] == len(payload), resp.metadata
            assert "oversized" not in resp.metadata, resp.metadata
            await store.close()
        print("✓ test_read_image_png passed")

    _run(run())


def test_read_image_jpeg():
    """``.jpg`` suffix maps to ``image/jpeg``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            payload = b"\xff\xd8\xff\xe0" + b"jpeg-body"
            _seed_bytes("dog.jpg", payload)
            store = await _make_store()
            resp = await _read_image(store, path="dog.jpg")
            assert resp.success is True, resp
            assert base64.b64decode(resp.answer) == payload
            assert resp.metadata["mime"] == "image/jpeg", resp.metadata
            await store.close()
        print("✓ test_read_image_jpeg passed")

    _run(run())


def test_read_image_oversized():
    """Above ``max_bytes`` → ``answer`` is a notice, ``metadata.oversized=True``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            payload = b"\x89PNG\r\n\x1a\n" + b"x" * 2048
            _seed_bytes("big.png", payload)
            store = await _make_store()
            resp = await _read_image(store, step_kwargs={"max_bytes": 1024}, path="big.png")
            assert resp.success is True, resp
            assert resp.metadata["oversized"] is True, resp.metadata
            assert resp.metadata["max_bytes"] == 1024, resp.metadata
            assert resp.metadata["size_bytes"] == len(payload), resp.metadata
            assert resp.metadata["mime"] == "image/png", resp.metadata
            try:
                decoded = base64.b64decode(resp.answer, validate=True)
                assert decoded != payload, "oversized branch must not return real base64"
            except Exception:
                pass  # expected — answer is notice text, not base64
            assert "exceeds max_bytes" in resp.answer
            await store.close()
        print("✓ test_read_image_oversized passed")

    _run(run())


def test_read_image_unknown_suffix():
    """Unknown suffix → still returns base64, ``metadata.non_image_warning=True``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            payload = b"any-bytes-here-for-blob"
            _seed_bytes("blob.xyz", payload)
            store = await _make_store()
            resp = await _read_image(store, path="blob.xyz")
            assert resp.success is True, resp
            assert base64.b64decode(resp.answer) == payload
            assert resp.metadata["non_image_warning"] is True, resp.metadata
            assert resp.metadata["mime"] is None, resp.metadata
            await store.close()
        print("✓ test_read_image_unknown_suffix passed")

    _run(run())


def test_read_image_no_suffix():
    """No suffix → compatibility mode (no auto-append), still reads as base64."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            payload = b"\x89PNG\r\n\x1a\nbody"
            _seed_bytes("no_suffix_blob", payload)
            store = await _make_store()
            resp = await _read_image(store, path="no_suffix_blob")
            assert resp.success is True, resp
            assert resp.metadata["non_image_warning"] is True, resp.metadata
            assert resp.metadata["mime"] is None, resp.metadata
            assert base64.b64decode(resp.answer) == payload
            await store.close()
        print("✓ test_read_image_no_suffix passed")

    _run(run())


def test_read_image_missing():
    """Non-existent path → ``success=False`` with ``does not exist`` message."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _read_image(store, path="never_existed.png")
            assert resp.success is False, resp
            assert resp.answer.startswith("Error:"), resp
            assert "does not exist" in resp.answer
            await store.close()
        print("✓ test_read_image_missing passed")

    _run(run())


def test_read_image_is_directory():
    """Path pointing to a directory → ``success=False`` with ``is not a file``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            (Path(tmp) / "subdir").mkdir(parents=True, exist_ok=True)
            store = await _make_store()
            resp = await _read_image(store, path="subdir")
            assert resp.success is False, resp
            assert "is not a file" in resp.answer, resp
            await store.close()
        print("✓ test_read_image_is_directory passed")

    _run(run())


def test_read_image_path_required():
    """Empty ``path`` → ``success=False`` with ``path is required`` message."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _read_image(store, path="")
            assert resp.success is False, resp
            assert "`path` is required" in resp.answer, resp
            await store.close()
        print("✓ test_read_image_path_required passed")

    _run(run())


def test_read_image_invalid_max_bytes():
    """``max_bytes=-1`` → ``success=False`` with positive-integer error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_bytes("a.png", b"\x89PNG\r\n\x1a\nx")
            store = await _make_store()
            resp = await _read_image(store, step_kwargs={"max_bytes": -1}, path="a.png")
            assert resp.success is False, resp
            assert "positive integer" in resp.answer, resp
            await store.close()
        print("✓ test_read_image_invalid_max_bytes passed")

    _run(run())
