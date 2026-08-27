"""Tests for AutoImageStep: image resource files become caption daily notes.

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
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import frontmatter
import pytest
from agentscope.model import ChatModelBase
from PIL import Image

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.file_store import LocalFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.evolve.auto_image import AutoImageStep, _parse_caption_json
from reme.steps.evolve.auto_resource import AutoResourceStep
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

    async def reply(self, inputs, **kwargs) -> dict:
        self.inputs = inputs
        return {"result": "ok"}


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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=_FakeVisionModel(fenced))
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])
                assert resp.success is True
                fenced_post = frontmatter.loads((cwd / "daily" / "2026-01-01" / "fenced-note.md").read_text("utf-8"))
                assert "Fenced caption body." in fenced_post.content

                raw = _write_binary(cwd / "resource" / "2026-01-01" / "photo.png", _png_bytes(color=(30, 30, 200)))
                step = AutoImageStep(
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=_FakeVisionModel("{}"))
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                result = resp.metadata["results"][0]
                assert resp.success is True
                assert result["metadata"]["reason"] == "vision_model_not_configured"
                assert not (cwd / "daily" / "2026-01-01" / "img.md").exists()
            finally:
                await fs.close()

    asyncio.run(run())


def test_image_and_text_changes_are_routed_by_suffix():
    """auto_image skips text changes; auto_resource skips image changes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            _install_file_jobs(app_ctx, fs)
            try:
                image_step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=_FakeVisionModel("{}"))
                resp = await _run_step(
                    image_step,
                    [{"change": "added", "path": str(cwd / "resource" / "2026-01-01" / "note.txt")}],
                )
                assert resp.metadata["results"][0]["metadata"]["reason"] == "non_image_file"

                wrapper = _FakeAgentWrapper()
                resource_step = AutoResourceStep(app_context=app_ctx, file_store=fs, agent_wrapper=wrapper)
                resp = await _run_step(
                    resource_step,
                    [{"change": "added", "path": str(cwd / "resource" / "2026-01-01" / "img.png")}],
                )
                assert resp.metadata["results"][0]["metadata"]["reason"] == "image_file"
                assert wrapper.inputs == ""
            finally:
                await fs.close()

    asyncio.run(run())


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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
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
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(source)}])

                assert resp.success is True
                assert len(model.plain_calls) == 1
                content = (cwd / "daily" / "2026-01-01" / "empty-note.md").read_text(encoding="utf-8")
                assert "Recovered by plain call." in content
            finally:
                await fs.close()

    asyncio.run(run())


def test_auto_image_converts_bmp_tiff_heic_requests():
    """bmp/tiff/heic resources are re-encoded for the request; notes record the source media_type."""

    async def run():
        pytest.importorskip("pillow_heif")
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
                    ("phone", "HEIF", ".heic"),
                ):
                    sources.append(_write_binary(cwd / "resource" / "2026-01-01" / f"{stem}{suffix}", _img_bytes(fmt)))
                model = _StructuredVisionModel(content={"name": "", "description": "d", "caption": "converted caption"})
                step = AutoImageStep(app_context=app_ctx, file_store=fs, as_llm=model)
                resp = await _run_step(step, [{"change": "added", "path": str(p)} for p in sources])

                assert resp.success is True
                assert len(model.structured_calls) == 3
                sent_mimes = set()
                for call in model.structured_calls:
                    sent_mimes.add(call[0].content[1].source.media_type)
                assert sent_mimes <= {"image/png", "image/jpeg"}
                for stem, expected_mime in (("photo", "image/bmp"), ("scan", "image/tiff"), ("phone", "image/heic")):
                    note = frontmatter.loads((cwd / "daily" / "2026-01-01" / f"{stem}.md").read_text(encoding="utf-8"))
                    assert note.metadata["media_type"] == expected_mime
                    assert "converted caption" in note.content
            finally:
                await fs.close()

    asyncio.run(run())
