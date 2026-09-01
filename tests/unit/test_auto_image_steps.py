"""Tests for AutoImageResourceStep: image resource files become caption daily notes.

The vision model boundary is faked (no network); test images are synthesized
with PIL inside a temporary workspace.
"""

# pylint: disable=protected-access

import asyncio
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import frontmatter
import pytest
import yaml
from agentscope.model import ChatModelBase
from PIL import Image

from reme.components import R
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.component_registry import ComponentRegistry
from reme.components.file_store import LocalFileStore
from reme.components.job import BaseJob
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
from reme.steps.evolve.base_auto_resource import BaseAutoResourceStep
from reme.steps.evolve.auto_image_resource import (
    AutoImageResourceStep,
    _build_image_request_payload,
    _normalize_image_bytes,
    _parse_caption_json,
)
from reme.steps.evolve.auto_resource import AutoResourceStep
from reme.steps.evolve.auto_text_resource import AutoTextResourceStep
from reme.steps.file_io import DailyListStep, FrontmatterUpdateStep, MoveStep, WriteStep


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


class _FakeAgentWrapper(BaseAgentWrapper):
    """Capture agent calls without invoking a real model."""

    def __init__(self):
        super().__init__()
        self.inputs = ""

    async def reply(self, inputs, **_kwargs) -> dict:
        """Capture the text processor input and return a successful result."""
        self.inputs = inputs
        return {"result": "ok"}


class _FlakyAgentWrapper(BaseAgentWrapper):
    """Fail one text item, then succeed so per-resource isolation is observable."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def reply(self, _inputs, **_kwargs) -> dict:
        """Raise on the first call and return normally on later calls."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("text provider unavailable")
        return {"result": "recovered"}


class _FakeVisionModel(ChatModelBase):
    """Capture VLM calls and return a canned text response (plain-call path)."""

    def __init__(self, text: str):
        self.text = text
        self.calls: list = []

    async def generate_structured_output(self, messages, structured_model, **kwargs):  # pylint: disable=unused-argument
        """Force the plain-call path in tests."""
        raise NotImplementedError("structured path not faked")

    async def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=[{"type": "text", "text": self.text}])


class _FakeAudioResourceStep(BaseAutoResourceStep):
    """Minimal third-modality processor used to verify the router extension contract."""

    resource_suffixes = frozenset({".wav"})

    async def _handle_upsert(self, file_path: str, date_str: str, note_stem: str, added: bool) -> None:
        self.context.response.success = True
        self.context.response.answer = f"Processed audio resource: {file_path}"
        self.context.response.metadata.update(
            {
                "path": f"daily/{date_str}/{note_stem}.md",
                "action": "added" if added else "modified",
                "processor": "audio",
                "modified": True,
            },
        )


class _FlakyVisionModel(ChatModelBase):
    """Fail the first plain call, succeed afterwards."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def generate_structured_output(self, messages, structured_model, **kwargs):  # pylint: disable=unused-argument
        """Force the plain-call path in tests."""
        raise NotImplementedError("structured path not faked")

    async def __call__(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("vision backend unavailable")
        return SimpleNamespace(content=[{"type": "text", "text": self.text}])


class _StructuredVisionModel(ChatModelBase):
    """Serve the schema-forced structured path; count fallback plain calls."""

    def __init__(self, content: dict | None = None, error: Exception | None = None, plain_text: str = "plain"):
        self.content = content
        self.error = error
        self.plain_text = plain_text
        self.structured_calls: list = []
        self.plain_calls: list = []

    async def generate_structured_output(self, messages, structured_model, **kwargs):  # pylint: disable=unused-argument
        """Serve the canned structured content (or raise the canned error)."""
        self.structured_calls.append(messages)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=dict(self.content or {}))

    async def __call__(self, messages, **kwargs):  # pylint: disable=unused-argument
        """Serve the canned plain-call text response."""
        self.plain_calls.append(messages)
        return SimpleNamespace(content=[{"type": "text", "text": self.plain_text}])


class _StepJob:
    """Tiny job adapter for unit tests that need BaseStep.run_job."""

    def __init__(self, step_cls, app_context, file_store):
        self.step_cls = step_cls
        self.app_context = app_context
        self.file_store = file_store

    async def __call__(self, **kwargs):
        step = self.step_cls(app_context=self.app_context, file_store=self.file_store)
        result = await step(**kwargs)
        return result or step.context.response


def _make_app_context(workspace_path: Path):
    """Create a mock app_context with app_config pointing to the given workspace."""
    ctx = MagicMock()
    ctx.app_config.workspace_dir = str(workspace_path)
    ctx.app_config.daily_dir = "daily"
    ctx.app_config.digest_dir = "digest"
    ctx.app_config.resource_dir = "resource"
    ctx.app_config.session_dir = "session"
    ctx.app_config.timezone = None
    return ctx


def _install_file_jobs(app_context, file_store) -> None:
    app_context.jobs = {
        "daily_list": _StepJob(DailyListStep, app_context, file_store),
        "frontmatter_update": _StepJob(FrontmatterUpdateStep, app_context, file_store),
        "move": _StepJob(MoveStep, app_context, file_store),
        "write": _StepJob(WriteStep, app_context, file_store),
    }


def _png_bytes(width: int = 8, height: int = 8, color=(200, 30, 30)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _img_bytes(image_format: str, size=(8, 8), color=(200, 30, 30)) -> bytes:
    """Synthesize an image in any PIL-supported format (incl. HEIF via pillow-heif)."""
    if image_format == "HEIF":
        from pillow_heif import register_heif_opener

        register_heif_opener()
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _write_binary(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _write_note(path: Path, source_resource: str, body: str = "old caption") -> Path:
    content = (
        f"---\nname: {path.stem}\ndescription: old\n"
        f'source_resource: "{source_resource}"\nkind: image\n'
        f"media_type: image/png\n---\n![[{source_resource[2:-2]}]]\n\n## Caption\n\n{body}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _caption_json(name: str, description: str, caption: str) -> str:
    return json.dumps({"name": name, "description": description, "caption": caption})


def _run_step(step, changes, **context_kwargs):
    return step(RuntimeContext(changes=changes, **context_kwargs))


def test_auto_image_creates_caption_note():
    """An added image produces a renamed daily note with caption and embed link."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _FakeVisionModel(_caption_json("red-square", "A red square", "An 8x8 solid red square."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                note_path = cwd / "daily" / "2026-01-01" / "red-square.md"
                assert note_path.is_file()
                post = frontmatter.loads(note_path.read_text(encoding="utf-8"))
                assert post.metadata["source_resource"] == "[[resource/2026-01-01/img.png]]"
                assert post.metadata["kind"] == "image"
                assert post.metadata["media_type"] == "image/png"
                assert post.metadata["name"] == "red-square"
                assert "![[resource/2026-01-01/img.png]]" in post.content
                assert "An 8x8 solid red square." in post.content
                assert (cwd / "daily" / "2026-01-01.md").is_file()
                assert len(model.calls) == 1
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_parses_fenced_json_and_falls_back_to_raw_text():
    """Fenced JSON is parsed; non-JSON output degrades to a raw-text caption."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                fenced = "```json\n" + _caption_json("fenced-note", "Fenced", "Fenced caption body.") + "\n```"
                source = _write_binary(cwd / "resource" / "2026-01-01" / "fenced.png", _png_bytes())
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=_FakeVisionModel(fenced))
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])
                assert resp.success is True
                fenced_post = frontmatter.loads((cwd / "daily" / "2026-01-01" / "fenced-note.md").read_text("utf-8"))
                assert "Fenced caption body." in fenced_post.content

                raw = _write_binary(cwd / "resource" / "2026-01-01" / "photo.png", _png_bytes(color=(30, 30, 200)))
                step = AutoImageResourceStep(
                    app_context=app_ctx,
                    file_store=fs,
                    as_llm=_FakeVisionModel("A plain description."),
                )
                resp = await _run_step(step, [{"change": "added", "path": str(raw)}])
                assert resp.success is True
                raw_post = frontmatter.loads((cwd / "daily" / "2026-01-01" / "photo.md").read_text("utf-8"))
                assert "A plain description." in raw_post.content
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_updates_existing_note_in_place():
    """A modified image rewrites the same note found via source_resource."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                note_path = _write_note(
                    cwd / "daily" / "2026-01-01" / "red-square.md",
                    "[[resource/2026-01-01/img.png]]",
                )
                model = _FakeVisionModel(_caption_json("red-square", "Updated", "The updated caption."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "modified", "path": str(source)}])

                assert resp.success is True
                assert note_path.is_file()
                post = frontmatter.loads(note_path.read_text(encoding="utf-8"))
                assert "The updated caption." in post.content
                assert not (cwd / "daily" / "2026-01-01" / "img.md").exists()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_deletes_linked_note():
    """Deleting the image resource removes its caption note."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = cwd / "resource" / "2026-01-01" / "img.png"
                note_path = _write_note(
                    cwd / "daily" / "2026-01-01" / "red-square.md",
                    "[[resource/2026-01-01/img.png]]",
                )
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=_FakeVisionModel("{}"))
                resp = await _run_step(step, [{"change": "deleted", "path": str(source)}])

                assert resp.success is True
                assert not note_path.exists()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_downscales_oversized_image_for_request_only():
    """Images beyond the request budget are downscaled in the request; storage is untouched."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(
                    cwd / "resource" / "2026-01-01" / "huge.png",
                    _png_bytes(width=3000, height=3000),
                )
                stored_bytes = source.read_bytes()
                model = _FakeVisionModel(_caption_json("huge-image", "Big", "A big image."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                assert len(model.calls) == 1
                data_block = model.calls[0][0].content[1]
                assert data_block.source.media_type == "image/jpeg"
                with Image.open(io.BytesIO(base64.b64decode(data_block.source.data))) as sent:
                    assert max(sent.size) <= 2048
                assert source.read_bytes() == stored_bytes
                assert (cwd / "daily" / "2026-01-01" / "huge-image.md").is_file()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_skips_oversized_file():
    """Files beyond max_image_bytes are skipped without a VLM call."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _FakeVisionModel(_caption_json("x", "y", "z"))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}], max_image_bytes=8)

                result = resp.metadata["results"][0]
                assert resp.success is True
                assert result["metadata"]["reason"] == "file_too_large"
                assert result["metadata"]["oversized"] is True
                assert not model.calls
                assert not (cwd / "daily" / "2026-01-01" / "img.md").exists()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_skips_without_vision_model():
    """Without any resolvable vision model the change is skipped with a reason."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            app_ctx.components = {}
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                result = resp.metadata["results"][0]
                assert resp.success is True
                assert result["metadata"]["reason"] == "vision_model_not_configured"
                assert not (cwd / "daily" / "2026-01-01" / "img.md").exists()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_resource_router_preserves_mixed_result_order_and_emits_one_hook():
    """The unified router sends each suffix to one processor and aggregates once."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            app_ctx.registry = R.copy()
            app_ctx.registry.add("fake_audio_resource_step", _FakeAudioResourceStep, owner=__name__)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                image = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                text = cwd / "resource" / "2026-01-01" / "note.txt"
                text.parent.mkdir(parents=True, exist_ok=True)
                text.write_text("hello", encoding="utf-8")
                wrapper = _FakeAgentWrapper()
                model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
                hook_calls = []

                async def hook(**kwargs):
                    hook_calls.append(kwargs)

                app_ctx.metadata = {"qwenpaw_memory_result_hook": hook}
                changes = [
                    {"change": "added", "path": str(image)},
                    {"change": "added", "path": str(cwd / "resource" / "2026-01-01" / "clip.wav")},
                    {"change": "added", "path": str(text)},
                ]
                step = AutoResourceStep(
                    app_context=app_ctx,
                    file_store=fs,
                    agent_wrapper=wrapper,
                    as_llm=model,
                    dispatch_steps=[
                        "auto_image_resource_step",
                        "fake_audio_resource_step",
                        "auto_text_resource_step",
                    ],
                )
                context = RuntimeContext(changes=changes)
                resp = await step(context)

                assert resp.success is True
                assert [item["path"] for item in resp.metadata["results"]] == [
                    "resource/2026-01-01/img.png",
                    "resource/2026-01-01/clip.wav",
                    "resource/2026-01-01/note.txt",
                ]
                assert all(
                    (item.get("metadata") or {}).get("reason") not in {"image_file", "non_image_file"}
                    for item in resp.metadata["results"]
                )
                assert len(model.calls) == 1
                assert "hello" in wrapper.inputs
                assert context.get("changes") == changes
                assert len(hook_calls) == 1
                assert hook_calls[0]["kwargs"] == {"changes": changes}
                assert len(hook_calls[0]["metadata"]["results"]) == 3
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_resource_router_does_not_mask_text_failure_with_image_success():
    """A later successful image result cannot overwrite an earlier text failure."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            app_ctx.registry = R
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                missing_text = cwd / "resource" / "2026-01-01" / "missing.txt"
                image = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
                changes = [
                    {"change": "added", "path": str(missing_text)},
                    {"change": "added", "path": str(image)},
                ]
                step = AutoResourceStep(
                    app_context=app_ctx,
                    dispatch_steps=[
                        {"backend": "auto_image_resource_step", "file_store": fs, "as_llm": model},
                        {
                            "backend": "auto_text_resource_step",
                            "file_store": fs,
                            "agent_wrapper": _FakeAgentWrapper(),
                        },
                    ],
                )
                resp = await step(RuntimeContext(changes=changes))

                assert resp.success is False
                assert resp.metadata["results"][0]["success"] is False
                assert "Resource file not found" in resp.metadata["results"][0]["answer"]
                assert resp.metadata["results"][1]["success"] is True
                assert (cwd / "daily" / "2026-01-01" / "red-square.md").is_file()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_resource_router_isolates_text_exception_and_preserves_image_result():
    """One text exception cannot discard an earlier image write or stop later resources."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            app_ctx.registry = R.copy()
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                image = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                first_text = cwd / "resource" / "2026-01-01" / "first.txt"
                second_text = cwd / "resource" / "2026-01-01" / "second.txt"
                first_text.write_text("first", encoding="utf-8")
                second_text.write_text("second", encoding="utf-8")
                model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
                wrapper = _FlakyAgentWrapper()
                hook_calls = []

                async def hook(**kwargs):
                    hook_calls.append(kwargs)

                app_ctx.metadata = {"qwenpaw_memory_result_hook": hook}
                changes = [
                    {"change": "added", "path": str(first_text)},
                    {"change": "added", "path": str(image)},
                    {"change": "added", "path": str(second_text)},
                ]
                context = RuntimeContext(changes=changes)
                step = AutoResourceStep(
                    app_context=app_ctx,
                    dispatch_steps=[
                        {"backend": "auto_image_resource_step", "file_store": fs, "as_llm": model},
                        {
                            "backend": "auto_text_resource_step",
                            "file_store": fs,
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
                assert (cwd / "daily" / "2026-01-01" / "red-square.md").is_file()
                assert context.get("changes") == changes
                assert len(hook_calls) == 1
                assert hook_calls[0]["metadata"]["results"] == results
            finally:
                await fs.close()

    asyncio.run(run())


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


def test_auto_resource_router_accepts_a_registered_third_modality_without_code_changes():
    """A new processor only needs registration, a matcher, and dispatch configuration."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            app_ctx.registry = R.copy()
            app_ctx.registry.add("fake_audio_resource_step", _FakeAudioResourceStep, owner=__name__)
            step = AutoResourceStep(
                app_context=app_ctx,
                dispatch_steps=["fake_audio_resource_step", "auto_text_resource_step"],
            )
            response = await step(
                RuntimeContext(
                    changes=[{"change": "added", "path": str(cwd / "resource" / "2026-01-01" / "clip.WAV")}],
                ),
            )

            assert response.success is True
            assert response.metadata["results"][0]["metadata"]["processor"] == "audio"
            assert response.metadata["results"][0]["path"] == "resource/2026-01-01/clip.WAV"

            unsupported = AutoResourceStep(
                app_context=app_ctx,
                dispatch_steps=["fake_audio_resource_step"],
            )
            response = await unsupported(
                RuntimeContext(
                    changes=[{"change": "added", "path": "resource/2026-01-01/archive.bin"}],
                ),
            )

            assert response.success is False
            assert response.metadata["results"][0]["metadata"]["reason"] == "unsupported_resource"

    asyncio.run(run())


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


def test_auto_resource_legacy_job_config_dispatches_text_resource():
    """A real BaseJob accepts the pre-router auto_resource Step configuration."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            app_ctx.registry = R.copy()
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            wrapper = _FakeAgentWrapper()
            job = BaseJob(
                name="legacy_auto_resource",
                app_context=app_ctx,
                steps=[
                    {
                        "backend": "auto_resource_step",
                        "file_store": fs,
                        "agent_wrapper": wrapper,
                    },
                ],
            )
            await job.start()
            try:
                source = cwd / "resource" / "2026-01-01" / "legacy.txt"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("legacy text resource", encoding="utf-8")

                response = await job(changes=[{"change": "added", "path": str(source)}])

                assert response.success is True
                assert response.metadata["processed"] == 1
                assert response.metadata["results"][0]["success"] is True
                assert "legacy text resource" in wrapper.inputs
            finally:
                await job.close()
                await fs.close()

    asyncio.run(run())


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


def test_image_preprocessing_reports_dependency_and_decode_errors():
    """Lazy image dependencies and corrupt bytes produce actionable errors."""
    real_import = __import__

    def import_without_pillow(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("blocked Pillow")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_pillow):
        with pytest.raises(RuntimeError, match=r"Pillow.*reme-ai\[core\]"):
            _normalize_image_bytes(_png_bytes(), ".png")

    def import_without_heif(name, *args, **kwargs):
        if name == "pillow_heif":
            raise ImportError("blocked pillow-heif")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_heif):
        assert _normalize_image_bytes(_png_bytes(), ".png") is None
        with pytest.raises(RuntimeError, match=r"pillow-heif.*reme-ai\[image-heif\]"):
            _normalize_image_bytes(_png_bytes(), ".heic")

    with pytest.raises(RuntimeError, match="Failed to decode image"):
        _build_image_request_payload(b"not an image", ".png")

    bmp_data = _img_bytes("BMP")
    with patch("PIL.Image.Image.save", side_effect=OSError("encoder failed")):
        with pytest.raises(RuntimeError, match="Failed to convert/resize image"):
            _normalize_image_bytes(bmp_data, ".bmp")


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


def test_auto_image_prompt_treats_filename_as_a_weak_hint():
    """The VLM prompt separates filename hints from visible image evidence."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "cat-at-beach.png", _png_bytes())
                model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                prompt = model.calls[0][0].content[0].text
                assert "Filename: cat-at-beach.png" in prompt
                assert "Filename stem: cat-at-beach" in prompt
                assert "weak hints" in prompt
                assert "trust the visible image content" in prompt
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_reports_modified_when_index_refresh_fails_after_write():
    """A post-write failure keeps the actual on-disk modification state."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _FakeVisionModel(_caption_json("red-square", "Red", "A red square."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)

                async def fail_refresh(*_args, **_kwargs):
                    raise RuntimeError("index refresh failed")

                with patch("reme.steps.evolve.base_auto_resource.refresh_day_index", new=fail_refresh):
                    resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                result = resp.metadata["results"][0]
                assert resp.success is False
                assert result["metadata"]["action"] == "failed"
                assert result["metadata"]["modified"] is True
                assert "index refresh failed" in result["metadata"]["error"]
                assert (cwd / "daily" / "2026-01-01" / "red-square.md").is_file()
            finally:
                await fs.close()

    asyncio.run(run())


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
    assert config["jobs"]["auto_image"]["steps"] == [{"backend": "auto_image_resource_step"}]


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


def test_auto_image_model_failure_is_isolated_per_change():
    """A failing VLM call marks one change failed while the rest of the batch continues."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                first = _write_binary(cwd / "resource" / "2026-01-01" / "img-a.png", _png_bytes())
                second = _write_binary(cwd / "resource" / "2026-01-01" / "img-b.png", _png_bytes(color=(20, 90, 200)))
                model = _FlakyVisionModel(_caption_json("blue-square", "Blue", "A blue square."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(
                    step,
                    [
                        {"change": "added", "path": str(first)},
                        {"change": "added", "path": str(second)},
                    ],
                )

                assert resp.success is False
                results = resp.metadata["results"]
                assert results[0]["success"] is False
                assert results[0]["metadata"]["action"] == "failed"
                assert results[1]["success"] is True
                assert (cwd / "daily" / "2026-01-01" / "blue-square.md").is_file()
                assert first.exists() and second.exists()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_decode_failure_is_isolated_per_change():
    """Corrupt image bytes fail one item without blocking a later valid image."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                corrupt = _write_binary(cwd / "resource" / "2026-01-01" / "bad.png", b"not an image")
                valid = _write_binary(cwd / "resource" / "2026-01-01" / "good.png", _png_bytes())
                model = _FakeVisionModel(_caption_json("good-image", "Good", "A valid image."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(
                    step,
                    [
                        {"change": "added", "path": str(corrupt)},
                        {"change": "added", "path": str(valid)},
                    ],
                )

                assert resp.success is False
                assert resp.metadata["results"][0]["metadata"]["action"] == "failed"
                assert "Failed to decode image" in resp.metadata["results"][0]["metadata"]["error"]
                assert resp.metadata["results"][1]["success"] is True
                assert len(model.calls) == 1
                assert (cwd / "daily" / "2026-01-01" / "good-image.md").is_file()
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_uniquifies_conflicting_note_name():
    """A name collision with an unrelated note falls back to the sha1-suffixed path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                _write_note(cwd / "daily" / "2026-01-01" / "red-square.md", "[[resource/2026-01-01/other.png]]")
                model = _FakeVisionModel(_caption_json("red-square", "A red square", "An 8x8 solid red square."))
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                suffix = hashlib.sha1(b"resource/2026-01-01/img.png").hexdigest()[:8]
                assert (cwd / "daily" / "2026-01-01" / f"red-square--{suffix}.md").is_file()
            finally:
                await fs.close()

    asyncio.run(run())


def test_parse_caption_json_cross_fills_missing_fields():
    """A description-only JSON payload cross-fills the caption instead of leaking raw JSON."""
    parsed = _parse_caption_json('{"description": "Waterfall in Iceland."}')
    assert parsed["caption"] == "Waterfall in Iceland."
    assert parsed["description"] == "Waterfall in Iceland."
    assert parsed["name"] == ""

    parsed = _parse_caption_json('{"caption": "A red square."}')
    assert parsed["caption"] == "A red square."
    assert parsed["description"] == ""

    parsed = _parse_caption_json('```json\n{"name": "n", "description": "d", "caption": "c"}\n```')
    assert parsed == {"name": "n", "description": "d", "caption": "c"}


def test_parse_caption_json_falls_back_to_raw_text():
    """Unusable payloads degrade to a raw-text caption."""
    parsed = _parse_caption_json("A plain description without json.")
    assert parsed == {"name": "", "description": "", "caption": "A plain description without json."}

    parsed = _parse_caption_json('{"foo": 1}')
    assert parsed["caption"] == '{"foo": 1}'


def test_auto_image_note_body_stays_clean_when_caption_field_missing():
    """Real-model regression: JSON with only a description must not enter the body verbatim."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _FakeVisionModel('{"file": "resource/2026-01-01/img.png", "description": "A tall waterfall."}')
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                note_path = cwd / "daily" / "2026-01-01" / "img.md"
                assert note_path.is_file()
                content = note_path.read_text(encoding="utf-8")
                assert "A tall waterfall." in content
                assert '{"file"' not in content
                assert '"description"' not in content
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_uses_structured_output_first():
    """The schema-forced structured call is the primary path; no plain fallback."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _StructuredVisionModel(
                    content={"name": "red-square", "description": "A red square.", "caption": "An 8x8 red square."},
                )
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                assert len(model.structured_calls) == 1
                assert not model.plain_calls
                note_path = cwd / "daily" / "2026-01-01" / "red-square.md"
                assert note_path.is_file()
                content = note_path.read_text(encoding="utf-8")
                assert "An 8x8 red square." in content
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_retries_with_plain_call_when_structured_fails():
    """A failing structured call retries once via the plain-call path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _StructuredVisionModel(
                    error=RuntimeError("provider rejects tool_choice"),
                    plain_text=_caption_json("plain-note", "Plain", "Plain-call caption."),
                )
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                assert len(model.structured_calls) == 1
                assert len(model.plain_calls) == 1
                note_path = cwd / "daily" / "2026-01-01" / "plain-note.md"
                content = note_path.read_text(encoding="utf-8")
                assert "Plain-call caption." in content
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_falls_back_when_structured_content_empty():
    """A structured response with no usable fields also triggers the plain retry."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(cwd / "resource" / "2026-01-01" / "img.png", _png_bytes())
                model = _StructuredVisionModel(
                    content={"name": "", "description": "", "caption": ""},
                    plain_text=_caption_json("empty-note", "Empty", "Recovered by plain call."),
                )
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                assert len(model.plain_calls) == 1
                content = (cwd / "daily" / "2026-01-01" / "empty-note.md").read_text(encoding="utf-8")
                assert "Recovered by plain call." in content
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_converts_bmp_tiff_requests():
    """Core BMP/TIFF conversions run without the optional HEIC dependency."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                sources = []
                for stem, fmt, suffix in (
                    ("photo", "BMP", ".bmp"),
                    ("scan", "TIFF", ".tiff"),
                ):
                    sources.append(_write_binary(cwd / "resource" / "2026-01-01" / f"{stem}{suffix}", _img_bytes(fmt)))
                model = _StructuredVisionModel(content={"name": "", "description": "d", "caption": "converted caption"})
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(p)} for p in sources])

                assert resp.success is True
                assert len(model.structured_calls) == 2
                sent_mimes = set()
                for call in model.structured_calls:
                    sent_mimes.add(call[0].content[1].source.media_type)
                assert sent_mimes <= {"image/png", "image/jpeg"}
                for stem, expected_mime in (("photo", "image/bmp"), ("scan", "image/tiff")):
                    note = frontmatter.loads((cwd / "daily" / "2026-01-01" / f"{stem}.md").read_text(encoding="utf-8"))
                    assert note.metadata["media_type"] == expected_mime
                    assert "converted caption" in note.content
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_converts_heic_request_when_extra_is_installed():
    """HEIC conversion is covered independently when the image-heif extra exists."""
    pytest.importorskip("pillow_heif")

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                source = _write_binary(
                    cwd / "resource" / "2026-01-01" / "phone.heic",
                    _img_bytes("HEIF"),
                )
                model = _StructuredVisionModel(
                    content={"name": "", "description": "d", "caption": "converted caption"},
                )
                step = AutoImageResourceStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                assert model.structured_calls[0][0].content[1].source.media_type in {"image/png", "image/jpeg"}
                note = frontmatter.loads((cwd / "daily" / "2026-01-01" / "phone.md").read_text(encoding="utf-8"))
                assert note.metadata["media_type"] == "image/heic"
                assert "converted caption" in note.content
            finally:
                await fs.close()

    asyncio.run(run())
