"""End-to-end tests for reme4 read_image step: spawn `reme4 start`, drive via HTTP.

Style mirrors test_crud_md_steps.py — temp working_dir, mock_reme_server,
call_and_check. The step reads an image file under working_dir, returning
base64 in `answer` plus path/size/mime in `metadata`.
"""

import asyncio
import base64
import os
import tempfile
import warnings
from pathlib import Path

from reme4.utils import call_action, mock_reme_server

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
    asyncio.run(coro)


def _seed_bytes(working_dir: Path, rel: str, data: bytes) -> Path:
    """Drop raw bytes at `working_dir/rel`. Step is byte-level, no need for real PNG."""
    target = working_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def test_read_image_png():
    """`reme4 read_image path=img/cat.png` returns base64 of file bytes + image/png mime."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            payload = b"\x89PNG\r\n\x1a\n" + b"fake-png-body-bytes"  # signature + body
            _seed_bytes(working, "img/cat.png", payload)

            async with mock_reme_server() as (host, port):
                result = await call_action("read_image", host=host, port=port, path="img/cat.png")

            assert result.get("success") is True, result
            assert base64.b64decode(result["answer"]) == payload, "base64 round-trip mismatch"
            md = result.get("metadata", {})
            assert md.get("mime") == "image/png", md
            assert md.get("size_bytes") == len(payload), md
            assert "oversized" not in md, md
        print("✓ test_read_image_png passed")

    _run(run())


def test_read_image_jpeg():
    """`.jpg` suffix maps to ``image/jpeg`` (also covers `.jpeg`)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            payload = b"\xff\xd8\xff\xe0" + b"jpeg-body"  # JPEG SOI marker + body
            _seed_bytes(working, "dog.jpg", payload)

            async with mock_reme_server() as (host, port):
                result = await call_action("read_image", host=host, port=port, path="dog.jpg")

            assert result.get("success") is True, result
            assert base64.b64decode(result["answer"]) == payload
            assert result["metadata"]["mime"] == "image/jpeg"
        print("✓ test_read_image_jpeg passed")

    _run(run())


def test_read_image_oversized():
    """Above ``max_bytes`` → ``answer`` is a notice (not base64), metadata.oversized=True."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            payload = b"\x89PNG\r\n\x1a\n" + b"x" * 2048
            _seed_bytes(working, "big.png", payload)

            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read_image",
                    host=host,
                    port=port,
                    path="big.png",
                    max_bytes=1024,  # force oversize
                )

            assert result.get("success") is True, result
            md = result.get("metadata", {})
            assert md.get("oversized") is True, md
            assert md.get("max_bytes") == 1024, md
            assert md.get("size_bytes") == len(payload), md
            assert md.get("mime") == "image/png", md
            # answer must NOT be valid base64 of the payload
            try:
                decoded = base64.b64decode(result["answer"], validate=True)
                # If it decoded, it certainly should not equal payload
                assert decoded != payload, "oversized branch must not return real base64"
            except Exception:
                pass  # expected — answer is notice text, not base64
            assert "exceeds max_bytes" in result["answer"]
        print("✓ test_read_image_oversized passed")

    _run(run())


def test_read_image_unknown_suffix():
    """Unknown suffix → still returns base64, but ``metadata.non_image_warning=True``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            payload = b"any-bytes-here-for-blob"
            _seed_bytes(working, "blob.xyz", payload)

            async with mock_reme_server() as (host, port):
                result = await call_action("read_image", host=host, port=port, path="blob.xyz")

            assert result.get("success") is True, result
            assert base64.b64decode(result["answer"]) == payload
            md = result.get("metadata", {})
            assert md.get("non_image_warning") is True, md
            assert md.get("mime") is None, md
        print("✓ test_read_image_unknown_suffix passed")

    _run(run())


def test_read_image_no_suffix():
    """No suffix → compatibility mode (no auto-append), still reads as base64."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            payload = b"\x89PNG\r\n\x1a\nbody"
            _seed_bytes(working, "no_suffix_blob", payload)

            async with mock_reme_server() as (host, port):
                result = await call_action("read_image", host=host, port=port, path="no_suffix_blob")

            assert result.get("success") is True, result
            md = result.get("metadata", {})
            assert md.get("non_image_warning") is True, md
            assert md.get("mime") is None, md
            assert base64.b64decode(result["answer"]) == payload
        print("✓ test_read_image_no_suffix passed")

    _run(run())


def test_read_image_missing():
    """Non-existent path → success=False with ``does not exist`` message."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read_image",
                    host=host,
                    port=port,
                    path="never_existed.png",
                )

            assert result.get("success") is False, result
            assert result["answer"].startswith("Error:"), result
            assert "does not exist" in result["answer"]
        print("✓ test_read_image_missing passed")

    _run(run())


def test_read_image_is_directory():
    """Path pointing to a directory → success=False with ``is not a file`` message."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            (working / "subdir").mkdir(parents=True, exist_ok=True)

            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read_image",
                    host=host,
                    port=port,
                    path="subdir",
                )

            assert result.get("success") is False, result
            assert "is not a file" in result["answer"], result
        print("✓ test_read_image_is_directory passed")

    _run(run())


def test_read_image_path_required():
    """Empty ``path`` → success=False with ``path is required`` message."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            (Path(tmp) / ".reme").mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action("read_image", host=host, port=port, path="")

            assert result.get("success") is False, result
            assert "`path` is required" in result["answer"], result
        print("✓ test_read_image_path_required passed")

    _run(run())


def test_read_image_invalid_max_bytes():
    """``max_bytes=-1`` (or non-int) → success=False with positive-integer error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_bytes(working, "a.png", b"\x89PNG\r\n\x1a\nx")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read_image",
                    host=host,
                    port=port,
                    path="a.png",
                    max_bytes=-1,
                )

            assert result.get("success") is False, result
            assert "positive integer" in result["answer"], result
        print("✓ test_read_image_invalid_max_bytes passed")

    _run(run())
