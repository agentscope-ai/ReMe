"""Tests for AutoImageResourceStep: image resource files become caption daily notes.

The agent reply boundary is faked (no network); real scoped file jobs write notes.
Test images are synthesized
with PIL inside a temporary workspace.
"""

# pylint: disable=protected-access

import base64
import hashlib
import io
import struct
import subprocess
import sys
import tomllib
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import frontmatter
import pytest
import yaml
from agentscope.message import DataBlock, TextBlock
from PIL import Image, JpegImagePlugin

from reme.components import R
from reme.components.component_registry import ComponentRegistry
from reme.components.job import BaseJob
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
from reme.steps.evolve.auto_image_resource import (
    AutoImageResourceStep,
    DEFAULT_MAX_IMAGE_PIXELS,
    _build_image_request_payload,
    _normalize_image_bytes,
)
from reme.steps.evolve.auto_resource import AutoResourceStep
from reme.steps.evolve.auto_text_resource import AutoTextResourceStep
from .auto_resource_test_support import (
    FakeAgentWrapper as _FakeAgentWrapper,
    FakeAudioResourceStep as _FakeAudioResourceStep,
    FakeImageAgentWrapper as _FakeImageAgentWrapper,
    FlakyAgentWrapper as _FlakyAgentWrapper,
    FlakyImageAgentWrapper as _FlakyImageAgentWrapper,
    caption_fields as _caption_fields,
    image_bytes as _img_bytes,
    make_app_context as _make_app_context,
    png_bytes as _png_bytes,
    write_binary as _write_binary,
    write_note as _write_note,
)

pytest_plugins = ("unit.auto_resource_test_plugin",)


def _png_bytes_with_header_size(width: int, height: int) -> bytes:
    """Change only a tiny PNG's IHDR dimensions without allocating its pixels."""
    data = bytearray(_png_bytes())
    data[16:24] = struct.pack(">II", width, height)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]) & 0xFFFFFFFF)
    return bytes(data)


@pytest.mark.parametrize(
    ("image_format", "suffix", "source_mime", "request_mime"),
    [
        ("PNG", ".png", "image/png", "image/png"),
        ("JPEG", ".jpg", "image/jpeg", "image/jpeg"),
        ("WEBP", ".webp", "image/webp", "image/webp"),
        ("BMP", ".bmp", "image/bmp", "image/jpeg"),
        ("TIFF", ".tiff", "image/tiff", "image/jpeg"),
        ("JPEG", ".png", "image/jpeg", "image/jpeg"),
        ("PNG", ".jpg", "image/png", "image/png"),
        ("BMP", ".png", "image/bmp", "image/jpeg"),
    ],
)
@pytest.mark.asyncio
async def test_auto_image_uses_decoded_format_for_request_and_note(
    image_format,
    suffix,
    source_mime,
    request_mime,
    auto_resource_env,
):
    """Normal or misleading suffixes preserve source bytes and use the actual decoded MIME."""
    env = auto_resource_env
    source = env.write_binary(f"resource/2026-01-01/image{suffix}", _img_bytes(image_format))
    stored_bytes = source.read_bytes()
    model = _FakeImageAgentWrapper(
        content={"name": "visible-subject", "description": "Visible", "caption": "Visible caption."},
    )
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    note_path = env.workspace / "daily/2026-01-01/visible-subject.md"
    post = frontmatter.loads(note_path.read_text(encoding="utf-8"))
    assert post.metadata["source_resource"] == f"[[resource/2026-01-01/image{suffix}]]"
    assert post.metadata["kind"] == "image"
    assert post.metadata["media_type"] == source_mime
    assert post.metadata["name"] == "visible-subject"
    assert f"![[resource/2026-01-01/image{suffix}]]" in post.content
    assert "Visible caption." in post.content
    data_block = next(block for block in model.calls[0][0].content if isinstance(block, DataBlock))
    assert data_block.source.media_type == request_mime
    with Image.open(io.BytesIO(base64.b64decode(data_block.source.data))) as sent:
        assert sent.get_format_mimetype() == request_mime
    assert source.read_bytes() == stored_bytes
    assert (env.workspace / "daily/2026-01-01.md").is_file()


@pytest.mark.parametrize("change", ["added", "modified", "deleted"])
@pytest.mark.asyncio
async def test_auto_image_note_lifecycle(change, auto_resource_env):
    """Added, modified, and deleted image events maintain one source-owned note."""
    env = auto_resource_env
    source = env.workspace / "resource/2026-01-01/img.png"
    note_path = env.workspace / "daily/2026-01-01/red-square.md"
    if change != "deleted":
        _write_binary(source, _png_bytes())
    if change != "added":
        _write_note(note_path, "[[resource/2026-01-01/img.png]]")

    model = _FakeImageAgentWrapper(_caption_fields("red-square", "Updated", "The updated caption."))
    response = await env.run(env.processor(model), [{"change": change, "path": str(source)}])

    result = response.metadata["results"][0]["metadata"]
    assert response.success is True
    assert result["action"] == change
    if change == "deleted":
        assert not note_path.exists()
        assert not model.calls
    else:
        assert note_path.is_file()
        assert "The updated caption." in frontmatter.loads(note_path.read_text(encoding="utf-8")).content


@pytest.mark.asyncio
async def test_auto_image_downscales_oversized_image_for_request_only(auto_resource_env):
    """Images beyond the request budget are downscaled in the request; storage is untouched."""

    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/huge.png", _png_bytes(width=3000, height=3000))
    stored_bytes = source.read_bytes()
    model = _FakeImageAgentWrapper(_caption_fields("huge-image", "Big", "A big image."))
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    data_block = next(block for block in model.calls[0][0].content if isinstance(block, DataBlock))
    assert data_block.source.media_type == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(data_block.source.data))) as sent:
        assert max(sent.size) <= 2048
    assert source.read_bytes() == stored_bytes
    assert (env.workspace / "daily/2026-01-01/huge-image.md").is_file()


@pytest.mark.asyncio
async def test_auto_image_uses_jpeg_decoder_downsampling_before_load(auto_resource_env):
    """Large JPEGs lower their decoder allocation before full pixel decode."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/large.jpg", _img_bytes("JPEG", (4096, 2048)))
    stored_bytes = source.read_bytes()
    model = _FakeImageAgentWrapper(_caption_fields("large-jpeg", "Large", "A large JPEG."))
    decoded_sizes = []
    original_load = JpegImagePlugin.JpegImageFile.load

    def record_decoder_size(image, *args, **kwargs):
        decoded_sizes.append(image.size)
        return original_load(image, *args, **kwargs)

    with patch.object(JpegImagePlugin.JpegImageFile, "load", new=record_decoder_size):
        response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    assert decoded_sizes
    assert decoded_sizes[0] == (2048, 1024)
    data_block = next(block for block in model.calls[0][0].content if isinstance(block, DataBlock))
    with Image.open(io.BytesIO(base64.b64decode(data_block.source.data))) as sent:
        assert max(sent.size) <= 2048
    assert source.read_bytes() == stored_bytes


def test_image_preprocessing_resizes_16_bit_tiff_before_rgb_conversion():
    """Pillow 10-compatible scaling keeps large I;16 TIFF images memory-bounded."""
    image = Image.new("I;16", (2049, 2), 1000)
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF")
    calls = []
    original_thumbnail = Image.Image.thumbnail

    def record_thumbnail(frame, size, resample, *args, **kwargs):
        calls.append((frame.mode, resample))
        return original_thumbnail(frame, size, resample, *args, **kwargs)

    with patch.object(Image.Image, "thumbnail", new=record_thumbnail):
        payload = _build_image_request_payload(buffer.getvalue(), ".tiff")

    assert calls == [("I;16", Image.Resampling.NEAREST)]
    assert payload["mime"] == "image/jpeg"
    assert payload["source_mime"] == "image/tiff"
    with Image.open(io.BytesIO(base64.b64decode(payload["data_b64"]))) as sent:
        assert max(sent.size) <= 2048


@pytest.mark.parametrize(("mode", "transparent"), [("P", False), ("1", False), ("P", True)])
def test_image_preprocessing_retains_thin_strokes_and_palette_alpha(mode, transparent):
    """Filtered downscaling retains lines that nearest-neighbor sampling drops."""
    image = Image.new(mode, (4096, 256), 1 if mode == "1" else 0)
    if mode == "P":
        image.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
        if transparent:
            image.info["transparency"] = 0
    image.paste(0 if mode == "1" else 1, (0, 0, 1, 256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()

    payload = _build_image_request_payload(buffer.getvalue(), ".png")

    assert payload["source_mime"] == "image/png"
    with Image.open(io.BytesIO(base64.b64decode(payload["data_b64"]))) as sent:
        assert sent.size == (2048, 128)
        if transparent:
            assert payload["mime"] == "image/png"
            assert sent.mode == "RGBA"
            alpha_min, alpha_max = sent.getextrema()[3]
            assert alpha_min == 0
            assert 0 < alpha_max < 255
        else:
            assert sent.getextrema()[0][0] < 240
            assert sent.getpixel((sent.width - 1, 0))[:3] == (255, 255, 255)


@pytest.mark.parametrize("mode", ["P", "1"])
@pytest.mark.parametrize("failing_method", ["thumbnail", "save"])
def test_image_preprocessing_closes_expanded_frames_on_failure(mode, failing_method):
    """Failure after mode expansion releases the temporary resize frame."""
    image = Image.new(mode, (2049, 2))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    converted_frames = []
    original_convert = Image.Image.convert

    def record_convert(frame, *args, **kwargs):
        converted = original_convert(frame, *args, **kwargs)
        if frame.mode == mode:
            converted_frames.append(converted)
        return converted

    with (
        patch.object(Image.Image, "convert", new=record_convert),
        patch.object(Image.Image, failing_method, side_effect=OSError(f"{failing_method} failed")),
        pytest.raises(RuntimeError, match="Failed to convert/resize image"),
    ):
        _build_image_request_payload(buffer.getvalue(), ".png")

    assert len(converted_frames) == 1
    for frame in converted_frames:
        with pytest.raises(ValueError, match="closed image"):
            frame.getpixel((0, 0))


@pytest.mark.parametrize(
    ("source_size", "request_size"),
    [((3000, 1000), (683, 2048)), ((4096, 1024), (512, 2048))],
    ids=["resize-and-rotate", "decoder-downsample-and-rotate"],
)
@pytest.mark.asyncio
async def test_auto_image_applies_exif_orientation_before_resizing(source_size, request_size, auto_resource_env):
    """A rotated phone JPEG is normalized upright for the VLM without touching its source."""
    env = auto_resource_env
    image = Image.new("RGB", source_size, (255, 0, 0))
    image.paste((0, 0, 255), (image.width // 2, 0, image.width, image.height))
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    image.close()
    source = env.write_binary("resource/2026-01-01/phone.jpg", buffer.getvalue())
    stored_bytes = source.read_bytes()
    model = _FakeImageAgentWrapper(_caption_fields("upright-phone-photo", "Upright", "An upright phone photo."))

    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    data_block = next(block for block in model.calls[0][0].content if isinstance(block, DataBlock))
    assert data_block.source.media_type == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(data_block.source.data))) as sent:
        assert sent.size == request_size
        assert sent.getexif().get(274) is None
        top = sent.getpixel((sent.width // 2, sent.height // 4))
        bottom = sent.getpixel((sent.width // 2, 3 * sent.height // 4))
        assert top[0] > 240 and top[2] < 15
        assert bottom[2] > 240 and bottom[0] < 15
    assert source.read_bytes() == stored_bytes
    assert (env.workspace / "daily/2026-01-01/upright-phone-photo.md").is_file()


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
@pytest.mark.asyncio
async def test_auto_image_rejects_pixel_bomb_before_decode_and_isolates_batch(routed, auto_resource_env):
    """A small compressed file with unsafe dimensions fails without stopping the next image."""
    env = auto_resource_env
    unsafe = env.write_binary(
        "resource/2026-01-01/pixel-bomb.png",
        _png_bytes_with_header_size(8000, 6000),
    )
    safe = env.write_binary("resource/2026-01-01/safe.png", _png_bytes())
    model = _FakeImageAgentWrapper(_caption_fields("safe-image", "Safe", "A safe image."))

    response = await env.run(
        env.processor(model, routed=routed),
        [
            {"change": "added", "path": str(unsafe)},
            {"change": "added", "path": str(safe)},
        ],
    )

    results = response.metadata["results"]
    assert response.success is False
    assert [item["success"] for item in results] == [False, True]
    assert results[0]["metadata"]["action"] == "failed"
    assert results[0]["metadata"]["modified"] is False
    assert f"48000000 > {DEFAULT_MAX_IMAGE_PIXELS}" in results[0]["metadata"]["error"]
    assert len(model.calls) == 1
    assert not (env.workspace / "daily/2026-01-01/pixel-bomb.md").exists()
    assert (env.workspace / "daily/2026-01-01/safe-image.md").is_file()


def test_image_pixel_limit_is_checked_before_full_decode():
    """The explicit pixel budget is enforced from image headers before ``load``."""
    with patch("PIL.PngImagePlugin.PngImageFile.load", side_effect=AssertionError("must not decode")) as image_load:
        with pytest.raises(RuntimeError, match=r"8x8=64 > 63"):
            _normalize_image_bytes(_png_bytes(), ".png", max_image_pixels=63)
    image_load.assert_not_called()


def test_image_decompression_bomb_warning_becomes_an_error(monkeypatch):
    """Pillow's warning-only bomb threshold becomes a reportable processor error."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 32)

    with pytest.raises(RuntimeError, match="decompression-bomb protection"):
        _build_image_request_payload(_png_bytes(), ".png")


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
@pytest.mark.asyncio
async def test_decompression_bomb_warning_is_isolated_per_change(routed, auto_resource_env, monkeypatch):
    """A Pillow bomb warning fails one resource while a safe batch peer still completes."""
    env = auto_resource_env
    warned = env.write_binary("resource/2026-01-01/warned.png", _png_bytes())
    safe = env.write_binary("resource/2026-01-01/safe.png", _png_bytes(width=4, height=4))
    model = _FakeImageAgentWrapper(_caption_fields("safe-image", "Safe", "A safe image."))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 32)

    response = await env.run(
        env.processor(model, routed=routed),
        [
            {"change": "added", "path": str(warned)},
            {"change": "added", "path": str(safe)},
        ],
    )

    results = response.metadata["results"]
    assert response.success is False
    assert [item["success"] for item in results] == [False, True]
    assert "decompression-bomb protection" in results[0]["metadata"]["error"]
    assert len(model.calls) == 1
    assert not (env.workspace / "daily/2026-01-01/warned.md").exists()
    assert (env.workspace / "daily/2026-01-01/safe-image.md").is_file()


def test_auto_image_pixel_limit_cannot_be_raised_by_runtime_context():
    """Request-scoped kwargs cannot relax the processor's deployment safety cap."""
    step = AutoImageResourceStep(max_image_pixels=63)
    step.context = RuntimeContext(max_image_pixels=DEFAULT_MAX_IMAGE_PIXELS)

    assert step._max_image_pixels() == 63


@pytest.mark.asyncio
async def test_auto_image_skips_oversized_file(auto_resource_env):
    """Files beyond max_image_bytes are skipped without a VLM call."""

    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    model = _FakeImageAgentWrapper(_caption_fields("x", "y", "z"))
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}], max_image_bytes=8)

    result = response.metadata["results"][0]
    assert response.success is True
    assert result["metadata"]["reason"] == "file_too_large"
    assert result["metadata"]["oversized"] is True
    assert not model.calls
    assert not (env.workspace / "daily/2026-01-01/img.md").exists()


@pytest.mark.asyncio
async def test_auto_image_fails_without_agent_wrapper(auto_resource_env):
    """Missing agent configuration cannot silently generate a text-only image note."""

    env = auto_resource_env
    env.app_context.components = {}
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    step = AutoImageResourceStep(app_context=env.app_context, file_store=env.file_store)
    response = await env.run(step, [{"change": "added", "path": str(source)}])

    result = response.metadata["results"][0]
    assert response.success is False
    assert result["metadata"]["action"] == "failed"
    assert not (env.workspace / "daily/2026-01-01/img.md").exists()


@pytest.mark.asyncio
async def test_auto_resource_router_preserves_mixed_result_order_and_emits_one_hook(auto_resource_env):
    """The unified router sends each suffix to one processor and aggregates once."""
    env = auto_resource_env
    env.app_context.registry = R.copy()
    env.app_context.registry.add("fake_audio_resource_step", _FakeAudioResourceStep, owner=__name__)
    image = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    text = env.workspace / "resource/2026-01-01/note.txt"
    text.write_text("hello", encoding="utf-8")
    wrapper = _FakeAgentWrapper()
    model = _FakeImageAgentWrapper(_caption_fields("red-square", "Red", "A red square."))
    model.app_context = env.app_context
    hook_calls = []

    async def hook(**kwargs):
        hook_calls.append(kwargs)

    env.app_context.metadata = {"qwenpaw_memory_result_hook": hook}
    changes = [
        {"change": "added", "path": str(image)},
        {"change": "added", "path": str(env.workspace / "resource/2026-01-01/clip.wav")},
        {"change": "added", "path": str(text)},
    ]
    step = AutoResourceStep(
        app_context=env.app_context,
        file_store=env.file_store,
        agent_wrapper=wrapper,
        dispatch_steps=[
            {"backend": "auto_image_resource_step", "agent_wrapper": model},
            "fake_audio_resource_step",
            "auto_text_resource_step",
        ],
    )
    context = RuntimeContext(changes=changes)
    response = await step(context)

    assert response.success is True
    assert [item["path"] for item in response.metadata["results"]] == [
        "resource/2026-01-01/img.png",
        "resource/2026-01-01/clip.wav",
        "resource/2026-01-01/note.txt",
    ]
    assert len(model.calls) == 1
    assert "hello" in wrapper.inputs
    assert context.get("changes") == changes
    assert len(hook_calls) == 1
    assert hook_calls[0]["kwargs"] == {"changes": changes}
    assert len(hook_calls[0]["metadata"]["results"]) == 3


@pytest.mark.asyncio
async def test_auto_resource_router_isolates_text_exception_and_preserves_image_result(auto_resource_env):
    """One text exception cannot discard an earlier image write or stop later resources."""

    env = auto_resource_env
    env.app_context.registry = R.copy()
    image = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    first_text = env.workspace / "resource/2026-01-01/first.txt"
    second_text = env.workspace / "resource/2026-01-01/second.txt"
    first_text.write_text("first", encoding="utf-8")
    second_text.write_text("second", encoding="utf-8")
    model = _FakeImageAgentWrapper(_caption_fields("red-square", "Red", "A red square."))
    wrapper = _FlakyAgentWrapper()
    model.app_context = env.app_context
    hook_calls = []

    async def hook(**kwargs):
        hook_calls.append(kwargs)

    env.app_context.metadata = {"qwenpaw_memory_result_hook": hook}
    changes = [
        {"change": "added", "path": str(first_text)},
        {"change": "added", "path": str(image)},
        {"change": "added", "path": str(second_text)},
    ]
    context = RuntimeContext(changes=changes)
    step = AutoResourceStep(
        app_context=env.app_context,
        dispatch_steps=[
            {"backend": "auto_image_resource_step", "file_store": env.file_store, "agent_wrapper": model},
            {
                "backend": "auto_text_resource_step",
                "file_store": env.file_store,
                "agent_wrapper": wrapper,
            },
        ],
    )
    response = await step(context)

    results = response.metadata["results"]
    assert response.success is False
    assert response.metadata["processed"] == 3
    assert response.metadata["modified"] is True
    assert [item["path"] for item in results] == [
        "resource/2026-01-01/first.txt",
        "resource/2026-01-01/img.png",
        "resource/2026-01-01/second.txt",
    ]
    assert [item["success"] for item in results] == [False, True, True]
    assert results[0]["metadata"]["modified"] is False
    assert results[0]["metadata"]["action"] == "failed"
    assert results[0]["metadata"]["error"] == "text provider unavailable"
    assert results[1]["metadata"]["modified"] is True
    assert "error" not in results[2]["metadata"]
    assert wrapper.calls == 2
    assert (env.workspace / "daily/2026-01-01/red-square.md").is_file()
    assert context.get("changes") == changes
    assert len(hook_calls) == 1
    assert hook_calls[0]["metadata"]["results"] == results


def test_auto_resource_router_inherits_declared_options_with_child_override():
    """Each processor selects inherited router options; explicit child values win."""
    file_store = object()
    agent_wrapper = object()
    prompt_dict = {"system_prompt": "legacy text prompt"}
    step = AutoResourceStep(
        file_store=file_store,
        agent_wrapper=agent_wrapper,
        include_images=False,
        language="zh",
        prompt_dict=prompt_dict,
        max_file_bytes=4,
        max_image_bytes=8,
        max_image_pixels=64,
        dispatch_steps=[
            {"backend": "auto_image_resource_step", "max_image_bytes": 32, "max_image_pixels": 128},
            {"backend": "auto_text_resource_step", "max_file_bytes": 16},
        ],
    )

    specs = {spec["backend"]: spec for spec, _, _ in step._processor_routes()}
    assert specs["auto_text_resource_step"] == {
        "backend": "auto_text_resource_step",
        "file_store": file_store,
        "agent_wrapper": agent_wrapper,
        "language": "zh",
        "prompt_dict": prompt_dict,
        "max_file_bytes": 16,
    }
    assert specs["auto_image_resource_step"] == {
        "backend": "auto_image_resource_step",
        "file_store": file_store,
        "agent_wrapper": agent_wrapper,
        "language": "zh",
        "prompt_dict": prompt_dict,
        "max_image_bytes": 32,
        "max_image_pixels": 128,
    }

    inherited = AutoResourceStep(
        max_image_pixels=64,
        dispatch_steps=["auto_image_resource_step", "auto_text_resource_step"],
    )
    inherited_specs = {spec["backend"]: spec for spec, _, _ in inherited._processor_routes()}
    assert inherited_specs["auto_image_resource_step"]["max_image_pixels"] == 64


@pytest.mark.asyncio
async def test_auto_resource_router_accepts_a_registered_third_modality_without_code_changes(auto_resource_env):
    """A new processor only needs registration, a matcher, and dispatch configuration."""

    env = auto_resource_env
    env.app_context.registry = R.copy()
    env.app_context.registry.add("fake_audio_resource_step", _FakeAudioResourceStep, owner=__name__)
    step = AutoResourceStep(
        app_context=env.app_context,
        dispatch_steps=["fake_audio_resource_step", "auto_text_resource_step"],
    )
    response = await env.run(
        step,
        [{"change": "added", "path": str(env.workspace / "resource/2026-01-01/clip.WAV")}],
    )

    assert response.success is True
    assert response.metadata["results"][0]["metadata"]["processor"] == "audio"
    assert response.metadata["results"][0]["path"] == "resource/2026-01-01/clip.WAV"

    unsupported = AutoResourceStep(app_context=env.app_context, dispatch_steps=["fake_audio_resource_step"])
    response = await env.run(unsupported, [{"change": "added", "path": "resource/2026-01-01/archive.bin"}])

    assert response.success is False
    assert response.metadata["results"][0]["metadata"]["reason"] == "unsupported_resource"


def test_auto_resource_router_requires_the_fallback_processor_to_be_last():
    """Processor ordering stays deterministic and first-match routing remains extensible."""
    step = AutoResourceStep(
        dispatch_steps=["auto_text_resource_step", "auto_image_resource_step"],
    )

    with pytest.raises(ValueError, match="fallback processor must be last"):
        step._processor_routes()


def test_auto_resource_router_discovers_unique_fallback_for_legacy_config():
    """An old Step spec without dispatch_steps keeps its text-resource behavior."""
    app_ctx = _make_app_context(Path.cwd())
    app_ctx.registry = R.copy()
    step = AutoResourceStep(app_context=app_ctx)

    routes = step._processor_routes()

    assert [(spec, step_cls) for spec, step_cls, _ in routes] == [
        ({"backend": "auto_text_resource_step"}, AutoTextResourceStep),
    ]


@pytest.mark.asyncio
async def test_auto_resource_legacy_job_config_dispatches_text_resource(auto_resource_env):
    """A real BaseJob accepts the pre-router auto_resource Step configuration."""

    env = auto_resource_env
    env.app_context.registry = R.copy()
    wrapper = _FakeAgentWrapper()
    job = BaseJob(
        name="legacy_auto_resource",
        app_context=env.app_context,
        steps=[
            {
                "backend": "auto_resource_step",
                "file_store": env.file_store,
                "agent_wrapper": wrapper,
            },
        ],
    )
    await job.start()
    try:
        source = env.workspace / "resource/2026-01-01/legacy.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("legacy text resource", encoding="utf-8")
        response = await job(changes=[{"change": "added", "path": str(source)}])

        assert response.success is True
        assert response.metadata["processed"] == 1
        assert response.metadata["results"][0]["success"] is True
        assert "legacy text resource" in wrapper.inputs
    finally:
        await job.close()


def test_auto_resource_router_rejects_ambiguous_registered_fallbacks():
    """Legacy discovery fails closed when plugins register multiple fallbacks."""
    app_ctx = _make_app_context(Path.cwd())
    app_ctx.registry = R.copy()
    app_ctx.registry.add("second_text_fallback", AutoTextResourceStep, owner=__name__)
    step = AutoResourceStep(app_context=app_ctx)

    with pytest.raises(RuntimeError, match="exactly one registered fallback resource processor"):
        step._processor_routes()

    explicit = AutoResourceStep(app_context=app_ctx, dispatch_steps=["auto_text_resource_step"])
    assert len(explicit._processor_routes()) == 1


def test_auto_resource_router_rejects_missing_registered_fallback():
    """Legacy discovery fails clearly when no fallback processor is installed."""
    app_ctx = _make_app_context(Path.cwd())
    app_ctx.registry = ComponentRegistry()
    step = AutoResourceStep(app_context=app_ctx)

    with pytest.raises(RuntimeError, match=r"fallback resource processor; found: none"):
        step._processor_routes()


def test_resource_processors_have_canonical_registrations_and_isolated_prompts():
    """Each modality owns one backend and loads only its module-local prompts."""
    assert R.get(ComponentEnum.STEP, "auto_resource_step") is AutoResourceStep
    assert R.get(ComponentEnum.STEP, "auto_text_resource_step") is AutoTextResourceStep
    assert R.get(ComponentEnum.STEP, "auto_image_resource_step") is AutoImageResourceStep
    assert R.get(ComponentEnum.STEP, "auto_image_step") is None

    text_step = AutoTextResourceStep()
    image_step = AutoImageResourceStep()
    assert text_step.prompt.has_prompt("system_prompt")
    assert text_step.prompt.has_prompt("user_message_create")
    assert not text_step.prompt.has_prompt("resource_instructions")
    assert image_step.prompt.has_prompt("resource_instructions")
    assert image_step.prompt.has_prompt("system_prompt")
    assert image_step.prompt.has_prompt("user_message_create")


def test_auto_image_named_wrapper_uses_standard_ref_resolution():
    """An explicit wrapper owns model selection; an unrelated vision model is ignored."""
    app_ctx = _make_app_context(Path.cwd())
    named = _FakeImageAgentWrapper("named")
    default = _FakeImageAgentWrapper("default")
    app_ctx.components = {
        ComponentEnum.AGENT_WRAPPER: {"my_agent": named, "default": default},
        ComponentEnum.AS_LLM: {"vision": SimpleNamespace(model=object())},
    }
    step = AutoImageResourceStep(app_context=app_ctx, agent_wrapper="my_agent")
    step.context = RuntimeContext()
    assert step.agent_wrapper is named
    implicit = AutoImageResourceStep(app_context=app_ctx)
    implicit.context = RuntimeContext()
    assert implicit.agent_wrapper is default
    missing = AutoImageResourceStep(app_context=app_ctx, agent_wrapper="missing_agent")
    missing.context = RuntimeContext()
    assert missing.agent_wrapper is None  # The standard wrapper Ref is optional.


@pytest.mark.parametrize(
    ("blocked_module", "payload", "suffix", "error_pattern"),
    [
        ("PIL", _png_bytes(), ".png", r"Pillow.*reme-ai\[core\]"),
        (
            "pillow_heif",
            b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic",
            ".heic",
            r"pillow-heif.*reme-ai\[image-heif\]",
        ),
    ],
)
def test_image_preprocessing_reports_dependency_errors(blocked_module, payload, suffix, error_pattern):
    """Lazy image dependencies produce actionable installation errors."""
    real_import = __import__

    def import_without_dependency(name, *args, **kwargs):
        if name == blocked_module or (blocked_module == "PIL" and name.startswith("PIL.")):
            raise ImportError(f"blocked {blocked_module}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_dependency):
        with pytest.raises(RuntimeError, match=error_pattern):
            _normalize_image_bytes(payload, suffix)


def test_misleading_heic_suffix_does_not_load_optional_dependency():
    """A core image named ``.heic`` is decoded by content without loading pillow-heif."""
    real_import = __import__

    def import_without_heif(name, *args, **kwargs):
        if name == "pillow_heif":
            raise AssertionError("pillow-heif must not be loaded for PNG bytes")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_heif):
        payload = _build_image_request_payload(_png_bytes(), ".heic")

    assert payload["mime"] == "image/png"
    assert payload["source_mime"] == "image/png"
    assert payload["converted"] is False


@pytest.mark.parametrize(
    ("payload", "suffix", "failing_method", "error_pattern"),
    [
        (b"not an image", ".png", None, "Failed to decode image"),
        (_png_bytes(width=2049, height=1), ".png", "thumbnail", "Failed to convert/resize image"),
        (_img_bytes("BMP"), ".bmp", "save", "Failed to convert/resize image"),
    ],
    ids=["decode", "resize", "conversion"],
)
def test_image_preprocessing_reports_data_errors(payload, suffix, failing_method, error_pattern):
    """Decode, resize, and conversion failures remain explicit."""
    if failing_method is None:
        with pytest.raises(RuntimeError, match=error_pattern):
            _build_image_request_payload(payload, suffix)
        return

    with patch(
        f"PIL.Image.Image.{failing_method}",
        side_effect=OSError(f"{failing_method} failed"),
    ) as failing_call:
        with pytest.raises(RuntimeError, match=error_pattern):
            _normalize_image_bytes(payload, suffix)
    failing_call.assert_called_once()


def test_auto_image_module_imports_without_pillow():
    """Importing the registered image Step does not eagerly require Pillow."""
    root = Path(__file__).resolve().parents[2]
    script = """
import builtins

real_import = builtins.__import__

def import_without_pillow(name, *args, **kwargs):
    if name == "PIL" or name.startswith("PIL."):
        raise ImportError("Pillow unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_pillow
import reme.steps.evolve.auto_image_resource
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
@pytest.mark.parametrize("legacy_template", [False, True], ids=["default-template", "legacy-template"])
@pytest.mark.asyncio
async def test_auto_image_prompt_merges_resource_instructions(language, existing, legacy_template, auto_resource_env):
    """Shared create/update prompts include localized image requirements exactly once."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/cat-at-beach.png"
    source = env.write_binary(source_path, _png_bytes())
    if existing:
        env.write_note("daily/2026-01-01/cat-at-beach.md", f"[[{source_path}]]")
    model = _FakeImageAgentWrapper(_caption_fields("red-square", "Red", "A red square."))
    step = env.processor(model, language=language)
    if legacy_template:
        suffix = "_zh" if language == "zh" else ""
        prompt_name = "user_message_update" if existing else "user_message_create"
        step = env.processor(
            model,
            language=language,
            prompt_dict={
                f"resource_instructions{suffix}": "Custom image instructions: {filename}",
                f"{prompt_name}{suffix}": step.get_prompt(prompt_name).replace("{resource_instructions}", ""),
            },
        )
    response = await env.run(step, [{"change": "modified" if existing else "added", "path": str(source)}])

    assert response.success is True
    message, options = model.calls[0]
    assert [type(block) for block in message.content] == [DataBlock, TextBlock]
    prompt = message.get_text_content()
    instructions = step.prompt_format(
        "resource_instructions",
        file_path=source_path,
        filename="cat-at-beach.png",
        stem="cat-at-beach",
        date="2026-01-01",
    )
    assert prompt.count(instructions) == 1
    if legacy_template:
        assert instructions == "Custom image instructions: cat-at-beach.png"
        assert prompt.endswith(instructions)
    else:
        assert ("weak hints" if language == "en" else "弱提示") in prompt
        assert ("trust the visible image content" if language == "en" else "以图像中的可见内容为准") in prompt
    assert options["injected_job_kwargs"]["_allowed_paths"][0] in prompt
    assert ("read path=" in prompt) is existing


@pytest.mark.asyncio
async def test_auto_image_reports_modified_when_index_refresh_fails_after_write(auto_resource_env):
    """A post-write failure keeps the actual on-disk modification state."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    model = _FakeImageAgentWrapper(_caption_fields("red-square", "Red", "A red square."))

    async def fail_refresh(*_args, **_kwargs):
        raise RuntimeError("index refresh failed")

    with patch("reme.steps.evolve.base_auto_resource.refresh_day_index", new=fail_refresh):
        response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    result = response.metadata["results"][0]
    assert response.success is False
    assert result["metadata"]["action"] == "failed"
    assert result["metadata"]["modified"] is True
    assert "index refresh failed" in result["metadata"]["error"]
    assert (env.workspace / "daily/2026-01-01/red-square.md").is_file()


def test_default_resource_watcher_dispatches_only_the_unified_router():
    """Both init and live resource producers call one auto-resource router."""
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "reme" / "config" / "default.yaml").read_text(encoding="utf-8"))
    steps = config["jobs"]["resource_watch_loop"]["steps"]
    expected_processors = [
        "auto_image_resource_step",
        "auto_text_resource_step",
    ]

    for producer in steps:
        backends = [item["backend"] for item in producer["dispatch_steps"]]
        assert backends == ["update_catalog_step", "auto_resource_step"]
        router = producer["dispatch_steps"][1]
        assert router["dispatch_steps"] == expected_processors

    auto_resource = config["jobs"]["auto_resource"]["steps"][0]
    assert auto_resource["backend"] == "auto_resource_step"
    assert auto_resource["dispatch_steps"] == expected_processors
    include_images = config["jobs"]["auto_resource"]["parameters"]["properties"]["include_images"]
    assert include_images["type"] == "boolean"
    assert include_images["default"] is True
    assert "auto_image" not in config["jobs"]


def test_image_dependencies_keep_heif_support_optional():
    """Pillow is core, while pillow-heif stays isolated in its opt-in extra."""
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base = [item.lower() for item in project["dependencies"]]
    optional = project["optional-dependencies"]

    assert not any(item.startswith("pillow") for item in base)
    assert any(item.lower().startswith("pillow>") for item in optional["core"])
    assert not any(item.lower().startswith("pillow-heif") for item in optional["core"])
    assert any(item.lower().startswith("pillow-heif") for item in optional["image-heif"])
    assert "reme-ai[image-heif]" in optional["full"]


@pytest.mark.parametrize("failure_stage", ["model", "decode"])
@pytest.mark.asyncio
async def test_auto_image_failure_is_isolated_per_change(failure_stage, auto_resource_env):
    """Model and decode failures do not block the next image in a batch."""
    env = auto_resource_env
    if failure_stage == "model":
        first_data = _png_bytes()
        model = _FlakyImageAgentWrapper(_caption_fields("good-image", "Good", "A valid image."))
        expected_error = "image agent unavailable"
    else:
        first_data = b"not an image"
        model = _FakeImageAgentWrapper(_caption_fields("good-image", "Good", "A valid image."))
        expected_error = "Failed to decode image"
    first = env.write_binary("resource/2026-01-01/first.png", first_data)
    second = env.write_binary("resource/2026-01-01/second.png", _png_bytes(color=(20, 90, 200)))

    response = await env.run(
        env.processor(model),
        [
            {"change": "added", "path": str(first)},
            {"change": "added", "path": str(second)},
        ],
    )

    results = response.metadata["results"]
    assert response.success is False
    assert results[0]["success"] is False
    assert results[0]["metadata"]["action"] == "failed"
    assert expected_error in results[0]["metadata"]["error"]
    assert results[1]["success"] is True
    assert (env.workspace / "daily/2026-01-01/good-image.md").is_file()
    assert first.exists() and second.exists()


@pytest.mark.asyncio
async def test_auto_image_uniquifies_conflicting_note_name(auto_resource_env):
    """A name collision with an unrelated note falls back to the sha1-suffixed path."""

    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    env.write_note("daily/2026-01-01/red-square.md", "[[resource/2026-01-01/other.png]]")
    model = _FakeImageAgentWrapper(_caption_fields("red-square", "A red square", "An 8x8 solid red square."))
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    suffix = hashlib.sha1(b"resource/2026-01-01/img.png").hexdigest()[:8]
    assert (env.workspace / f"daily/2026-01-01/red-square--{suffix}.md").is_file()


@pytest.mark.asyncio
async def test_auto_image_converts_heic_request_when_extra_is_installed(auto_resource_env):
    """HEIC conversion is covered independently when the image-heif extra exists."""
    pytest.importorskip("pillow_heif")

    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/phone.heic", _img_bytes("HEIF"))
    model = _FakeImageAgentWrapper(content={"name": "", "description": "d", "caption": "converted caption"})
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    data_block = next(block for block in model.calls[0][0].content if isinstance(block, DataBlock))
    assert data_block.source.media_type in {"image/png", "image/jpeg"}
    note = frontmatter.loads((env.workspace / "daily/2026-01-01/phone.md").read_text(encoding="utf-8"))
    assert note.metadata["media_type"] == "image/heic"
    assert "converted caption" in note.content
