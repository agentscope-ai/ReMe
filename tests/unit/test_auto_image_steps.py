"""Tests for AutoImageResourceStep: image resource files become caption daily notes.

The vision model boundary is faked (no network); test images are synthesized
with PIL inside a temporary workspace.
"""

# pylint: disable=protected-access

import base64
import hashlib
import io
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import frontmatter
import pytest
import yaml
from PIL import Image

from reme.components import R
from reme.components.component_registry import ComponentRegistry
from reme.components.job import BaseJob
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
from reme.steps.evolve.auto_image_resource import (
    AutoImageResourceStep,
    _build_image_request_payload,
    _normalize_image_bytes,
    _parse_caption_json,
)
from reme.steps.evolve.auto_resource import AutoResourceStep
from reme.steps.evolve.auto_text_resource import AutoTextResourceStep
from .auto_resource_test_support import (
    FakeAgentWrapper as _FakeAgentWrapper,
    FakeAudioResourceStep as _FakeAudioResourceStep,
    FakeVisionModel as _FakeVisionModel,
    FlakyAgentWrapper as _FlakyAgentWrapper,
    FlakyVisionModel as _FlakyVisionModel,
    StructuredVisionModel as _StructuredVisionModel,
    caption_json as _caption_json,
    image_bytes as _img_bytes,
    make_app_context as _make_app_context,
    png_bytes as _png_bytes,
    write_binary as _write_binary,
    write_note as _write_note,
)


@pytest.mark.parametrize(
    ("image_format", "suffix", "source_mime", "request_mime"),
    [
        ("PNG", ".png", "image/png", "image/png"),
        ("JPEG", ".jpg", "image/jpeg", "image/jpeg"),
        ("WEBP", ".webp", "image/webp", "image/webp"),
        ("BMP", ".bmp", "image/bmp", "image/jpeg"),
        ("TIFF", ".tiff", "image/tiff", "image/jpeg"),
    ],
)
@pytest.mark.asyncio
async def test_auto_image_supports_core_formats(
    image_format,
    suffix,
    source_mime,
    request_mime,
    auto_resource_env,
):
    """Core formats preserve source metadata and use a provider-safe request payload."""
    env = auto_resource_env
    source = env.write_binary(f"resource/2026-01-01/image{suffix}", _img_bytes(image_format))
    model = _StructuredVisionModel(
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
    assert model.structured_calls[0][0].content[1].source.media_type == request_mime
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

    model = _FakeVisionModel(_caption_json("red-square", "Updated", "The updated caption."))
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


@pytest.mark.parametrize(
    ("stem", "plain_text", "note_name", "caption", "raw_json_must_be_absent"),
    [
        (
            "fenced",
            "```json\n" + _caption_json("fenced-note", "Fenced", "Fenced caption body.") + "\n```",
            "fenced-note",
            "Fenced caption body.",
            False,
        ),
        ("photo", "A plain description.", "photo", "A plain description.", False),
        (
            "waterfall",
            '{"file": "resource/2026-01-01/waterfall.png", "description": "A tall waterfall."}',
            "waterfall",
            "A tall waterfall.",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_auto_image_plain_outputs_create_clean_notes(
    stem,
    plain_text,
    note_name,
    caption,
    raw_json_must_be_absent,
    auto_resource_env,
):
    """Fenced JSON, raw text, and description-only JSON remain valid plain fallbacks."""
    env = auto_resource_env
    source = env.write_binary(f"resource/2026-01-01/{stem}.png", _png_bytes())
    response = await env.run(env.processor(_FakeVisionModel(plain_text)), [{"change": "added", "path": str(source)}])

    assert response.success is True
    content = (env.workspace / f"daily/2026-01-01/{note_name}.md").read_text(encoding="utf-8")
    assert caption in content
    if raw_json_must_be_absent:
        assert '{"file"' not in content
        assert '"description"' not in content


@pytest.mark.asyncio
async def test_auto_image_downscales_oversized_image_for_request_only(auto_resource_env):
    """Images beyond the request budget are downscaled in the request; storage is untouched."""

    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/huge.png", _png_bytes(width=3000, height=3000))
    stored_bytes = source.read_bytes()
    model = _FakeVisionModel(_caption_json("huge-image", "Big", "A big image."))
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    data_block = model.calls[0][0].content[1]
    assert data_block.source.media_type == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(data_block.source.data))) as sent:
        assert max(sent.size) <= 2048
    assert source.read_bytes() == stored_bytes
    assert (env.workspace / "daily/2026-01-01/huge-image.md").is_file()


@pytest.mark.asyncio
async def test_auto_image_skips_oversized_file(auto_resource_env):
    """Files beyond max_image_bytes are skipped without a VLM call."""

    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    model = _FakeVisionModel(_caption_json("x", "y", "z"))
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}], max_image_bytes=8)

    result = response.metadata["results"][0]
    assert response.success is True
    assert result["metadata"]["reason"] == "file_too_large"
    assert result["metadata"]["oversized"] is True
    assert not model.calls
    assert not (env.workspace / "daily/2026-01-01/img.md").exists()


@pytest.mark.asyncio
async def test_auto_image_skips_without_vision_model(auto_resource_env):
    """Without any resolvable vision model the change is skipped with a reason."""

    env = auto_resource_env
    env.app_context.components = {}
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    step = AutoImageResourceStep(app_context=env.app_context, file_store=env.file_store)
    response = await env.run(step, [{"change": "added", "path": str(source)}])

    result = response.metadata["results"][0]
    assert response.success is True
    assert result["metadata"]["reason"] == "vision_model_not_configured"
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
    model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
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
        as_llm=model,
        dispatch_steps=["auto_image_resource_step", "fake_audio_resource_step", "auto_text_resource_step"],
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
    model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
    wrapper = _FlakyAgentWrapper()
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
            {"backend": "auto_image_resource_step", "file_store": env.file_store, "as_llm": model},
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
    assert results[0]["metadata"] == {
        "path": "resource/2026-01-01/first.txt",
        "modified": False,
        "action": "failed",
        "error": "text provider unavailable",
    }
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
    vision_model = object()
    prompt_dict = {"system_prompt": "legacy text prompt"}
    step = AutoResourceStep(
        file_store=file_store,
        agent_wrapper=agent_wrapper,
        as_llm=vision_model,
        language="zh",
        prompt_dict=prompt_dict,
        max_file_bytes=4,
        max_image_bytes=8,
        dispatch_steps=[
            {"backend": "auto_image_resource_step", "max_image_bytes": 32},
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
        "as_llm": vision_model,
        "language": "zh",
        "max_image_bytes": 32,
    }


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
    assert not text_step.prompt.has_prompt("user_message")
    assert image_step.prompt.has_prompt("user_message")
    assert not image_step.prompt.has_prompt("system_prompt")
    assert not image_step.prompt.has_prompt("user_message_create")


def test_auto_image_named_model_uses_standard_ref_resolution():
    """A configured ``as_llm`` component name is honored instead of ignored."""
    app_ctx = _make_app_context(Path.cwd())
    named = _FakeVisionModel("named")
    vision = _FakeVisionModel("vision")
    default = _FakeVisionModel("default")
    app_ctx.components = {
        ComponentEnum.AS_LLM: {
            "my_vlm": SimpleNamespace(model=named),
            "vision": SimpleNamespace(model=vision),
            "default": SimpleNamespace(model=default),
        },
    }
    step = AutoImageResourceStep(app_context=app_ctx, as_llm="my_vlm")
    step.context = RuntimeContext()

    assert step._vision_model() is named

    implicit = AutoImageResourceStep(app_context=app_ctx)
    implicit.context = RuntimeContext()
    assert implicit._vision_model() is vision

    missing = AutoImageResourceStep(app_context=app_ctx, as_llm="missing_vlm")
    missing.context = RuntimeContext()
    with pytest.raises(KeyError, match="missing_vlm"):
        missing._vision_model()


@pytest.mark.parametrize(
    ("blocked_module", "suffix", "error_pattern"),
    [
        ("PIL", ".png", r"Pillow.*reme-ai\[core\]"),
        ("pillow_heif", ".heic", r"pillow-heif.*reme-ai\[image-heif\]"),
    ],
)
def test_image_preprocessing_reports_dependency_errors(blocked_module, suffix, error_pattern):
    """Lazy image dependencies produce actionable installation errors."""
    real_import = __import__

    def import_without_dependency(name, *args, **kwargs):
        if name == blocked_module or (blocked_module == "PIL" and name.startswith("PIL.")):
            raise ImportError(f"blocked {blocked_module}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_dependency):
        with pytest.raises(RuntimeError, match=error_pattern):
            _normalize_image_bytes(_png_bytes(), suffix)


@pytest.mark.parametrize(
    ("payload", "suffix", "save_fails", "error_pattern"),
    [
        (b"not an image", ".png", False, "Failed to decode image"),
        (_img_bytes("BMP"), ".bmp", True, "Failed to convert/resize image"),
    ],
    ids=["decode", "conversion"],
)
def test_image_preprocessing_reports_data_errors(payload, suffix, save_fails, error_pattern):
    """Decode and conversion failures remain explicit."""
    if not save_fails:
        with pytest.raises(RuntimeError, match=error_pattern):
            _build_image_request_payload(payload, suffix)
        return

    with patch("PIL.Image.Image.save", side_effect=OSError("encoder failed")):
        with pytest.raises(RuntimeError, match=error_pattern):
            _normalize_image_bytes(payload, suffix)


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


@pytest.mark.asyncio
async def test_auto_image_prompt_treats_filename_as_a_weak_hint(auto_resource_env):
    """The VLM prompt separates filename hints from visible image evidence."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/cat-at-beach.png", _png_bytes())
    model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    prompt = model.calls[0][0].content[0].text
    assert "Filename: cat-at-beach.png" in prompt
    assert "Filename stem: cat-at-beach" in prompt
    assert "weak hints" in prompt
    assert "trust the visible image content" in prompt


@pytest.mark.asyncio
async def test_auto_image_reports_modified_when_index_refresh_fails_after_write(auto_resource_env):
    """A post-write failure keeps the actual on-disk modification state."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))

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

    for producer in steps:
        backends = [item["backend"] for item in producer["dispatch_steps"]]
        assert backends == ["update_catalog_step", "auto_resource_step"]
        router = producer["dispatch_steps"][1]
        assert router["dispatch_steps"] == ["auto_image_resource_step", "auto_text_resource_step"]

    auto_resource = config["jobs"]["auto_resource"]["steps"][0]
    assert auto_resource["backend"] == "auto_resource_step"
    assert auto_resource["dispatch_steps"] == ["auto_image_resource_step", "auto_text_resource_step"]
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
        model = _FlakyVisionModel(_caption_json("good-image", "Good", "A valid image."))
        expected_error = "vision backend unavailable"
    else:
        first_data = b"not an image"
        model = _FakeVisionModel(_caption_json("good-image", "Good", "A valid image."))
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
    model = _FakeVisionModel(_caption_json("red-square", "A red square", "An 8x8 solid red square."))
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    suffix = hashlib.sha1(b"resource/2026-01-01/img.png").hexdigest()[:8]
    assert (env.workspace / f"daily/2026-01-01/red-square--{suffix}.md").is_file()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"description": "Waterfall in Iceland."}',
            {"name": "", "description": "Waterfall in Iceland.", "caption": "Waterfall in Iceland."},
        ),
        ('{"caption": "A red square."}', {"name": "", "description": "", "caption": "A red square."}),
        (
            '```json\n{"name": "n", "description": "d", "caption": "c"}\n```',
            {"name": "n", "description": "d", "caption": "c"},
        ),
        (
            "A plain description without json.",
            {"name": "", "description": "", "caption": "A plain description without json."},
        ),
        ('{"foo": 1}', {"name": "", "description": "", "caption": ""}),
        ("```json\n\n```", {"name": "", "description": "", "caption": ""}),
    ],
)
def test_parse_caption_json(text, expected):
    """Plain fallback parsing normalizes useful fields without leaking unusable JSON."""
    assert _parse_caption_json(text) == expected


@pytest.mark.parametrize(
    ("structured_mode", "note_name", "caption", "expected_plain_calls"),
    [
        ("success", "red-square", "An 8x8 red square.", 0),
        ("error", "plain-note", "Plain-call caption.", 1),
        ("empty", "empty-note", "Recovered by plain call.", 1),
    ],
)
@pytest.mark.asyncio
async def test_auto_image_structured_output_and_plain_retry(
    structured_mode,
    note_name,
    caption,
    expected_plain_calls,
    auto_resource_env,
):
    """Structured success stays primary; structured errors and empties retry plain once."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/img.png", _png_bytes())
    if structured_mode == "success":
        model = _StructuredVisionModel(
            content={"name": note_name, "description": "A red square.", "caption": caption},
        )
    elif structured_mode == "error":
        model = _StructuredVisionModel(
            error=RuntimeError("provider rejects tool_choice"),
            plain_text=_caption_json(note_name, "Plain", caption),
        )
    else:
        model = _StructuredVisionModel(
            content={"name": "", "description": "", "caption": ""},
            plain_text=_caption_json(note_name, "Empty", caption),
        )

    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    assert len(model.structured_calls) == 1
    assert len(model.plain_calls) == expected_plain_calls
    content = (env.workspace / f"daily/2026-01-01/{note_name}.md").read_text(encoding="utf-8")
    assert caption in content


@pytest.mark.asyncio
async def test_auto_image_converts_heic_request_when_extra_is_installed(auto_resource_env):
    """HEIC conversion is covered independently when the image-heif extra exists."""
    pytest.importorskip("pillow_heif")

    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/phone.heic", _img_bytes("HEIF"))
    model = _StructuredVisionModel(content={"name": "", "description": "d", "caption": "converted caption"})
    response = await env.run(env.processor(model), [{"change": "added", "path": str(source)}])

    assert response.success is True
    assert model.structured_calls[0][0].content[1].source.media_type in {"image/png", "image/jpeg"}
    note = frontmatter.loads((env.workspace / "daily/2026-01-01/phone.md").read_text(encoding="utf-8"))
    assert note.metadata["media_type"] == "image/heic"
    assert "converted caption" in note.content
