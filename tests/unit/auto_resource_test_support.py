"""Shared test harness for auto-resource processor and router tests."""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest_asyncio
from agentscope.model import ChatModelBase
from PIL import Image

from reme.components import R
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.file_store import LocalFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.evolve.auto_image_resource import AutoImageResourceStep
from reme.steps.evolve.auto_resource import AutoResourceStep
from reme.steps.evolve.base_auto_resource import BaseAutoResourceStep
from reme.steps.file_io import DailyListStep, FrontmatterUpdateStep, MoveStep, WriteStep


class FakeAgentWrapper(BaseAgentWrapper):
    """Capture text-processor calls without invoking a real model."""

    def __init__(self):
        super().__init__()
        self.inputs = ""

    async def reply(self, inputs, **_kwargs) -> dict:
        """Record and accept one text-processor request."""
        self.inputs = inputs
        return {"result": "ok"}


class FlakyAgentWrapper(BaseAgentWrapper):
    """Fail one text item, then succeed."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def reply(self, _inputs, **_kwargs) -> dict:
        """Fail the first request and accept subsequent ones."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("text provider unavailable")
        return {"result": "recovered"}


class FakeVisionModel(ChatModelBase):
    """Capture VLM calls and return canned plain text."""

    def __init__(self, text: str):
        self.text = text
        self.calls: list = []

    async def generate_structured_output(self, messages, structured_model, **kwargs):  # pylint: disable=unused-argument
        """Force callers through the plain fallback."""
        raise NotImplementedError("structured path not faked")

    async def __call__(self, messages, **kwargs):
        """Record a call and return the canned plain text."""
        self.calls.append(messages)
        return SimpleNamespace(content=[{"type": "text", "text": self.text}])


class FlakyVisionModel(ChatModelBase):
    """Fail the first plain call, then succeed."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def generate_structured_output(self, messages, structured_model, **kwargs):  # pylint: disable=unused-argument
        """Force callers through the plain fallback."""
        raise NotImplementedError("structured path not faked")

    async def __call__(self, messages, **kwargs):
        """Fail once, then return the canned plain text."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("vision backend unavailable")
        return SimpleNamespace(content=[{"type": "text", "text": self.text}])


class StructuredVisionModel(ChatModelBase):
    """Serve structured output and count fallback plain calls."""

    def __init__(self, content: dict | None = None, error: Exception | None = None, plain_text: str = "plain"):
        self.content = content
        self.error = error
        self.plain_text = plain_text
        self.structured_calls: list = []
        self.plain_calls: list = []

    async def generate_structured_output(self, messages, structured_model, **kwargs):  # pylint: disable=unused-argument
        """Return or fail with the configured structured response."""
        self.structured_calls.append(messages)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=dict(self.content or {}))

    async def __call__(self, messages, **kwargs):  # pylint: disable=unused-argument
        """Record and return the configured plain fallback."""
        self.plain_calls.append(messages)
        return SimpleNamespace(content=[{"type": "text", "text": self.plain_text}])


class FakeAudioResourceStep(BaseAutoResourceStep):
    """Minimal third modality used to verify the router extension contract."""

    resource_suffixes = frozenset({".wav"})

    async def _handle_upsert(
        self,
        file_path: str,
        date_str: str,
        note_stem: str,
        added: bool,
        source_path: Path,
    ) -> None:
        del source_path
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


class _StepJob:
    """Tiny job adapter for tests that need ``BaseStep.run_job``."""

    def __init__(self, step_cls, app_context, file_store):
        self.step_cls = step_cls
        self.app_context = app_context
        self.file_store = file_store

    async def __call__(self, **kwargs):
        step = self.step_cls(app_context=self.app_context, file_store=self.file_store)
        result = await step(**kwargs)
        return result or step.context.response


def make_app_context(workspace: Path):
    """Create the minimal application context used by resource tests."""
    context = MagicMock()
    context.app_config.workspace_dir = str(workspace)
    context.app_config.daily_dir = "daily"
    context.app_config.digest_dir = "digest"
    context.app_config.resource_dir = "resource"
    context.app_config.session_dir = "session"
    context.app_config.timezone = None
    return context


def _install_file_jobs(app_context, file_store) -> None:
    app_context.jobs = {
        "daily_list": _StepJob(DailyListStep, app_context, file_store),
        "frontmatter_update": _StepJob(FrontmatterUpdateStep, app_context, file_store),
        "move": _StepJob(MoveStep, app_context, file_store),
        "write": _StepJob(WriteStep, app_context, file_store),
    }


def image_bytes(image_format: str = "PNG", size=(8, 8), color=(200, 30, 30)) -> bytes:
    """Synthesize a small image in a Pillow-supported format."""
    if image_format == "HEIF":
        from pillow_heif import register_heif_opener

        register_heif_opener()
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def png_bytes(width: int = 8, height: int = 8, color=(200, 30, 30)) -> bytes:
    """Compatibility shorthand for PNG-focused assertions."""
    return image_bytes("PNG", (width, height), color)


def write_binary(path: Path, data: bytes) -> Path:
    """Write test bytes, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_note(path: Path, source_resource: str, body: str = "old caption") -> Path:
    """Write a minimal source-owned image note."""
    content = (
        f"---\nname: {path.stem}\ndescription: old\n"
        f'source_resource: "{source_resource}"\nkind: image\n'
        f"media_type: image/png\n---\n![[{source_resource[2:-2]}]]\n\n## Caption\n\n{body}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def caption_json(name: str, description: str, caption: str) -> str:
    """Build a plain-call caption payload."""
    return json.dumps({"name": name, "description": description, "caption": caption})


def image_processor(app_context, file_store, model, *, routed: bool):
    """Build either the image processor or the public unified-router path."""
    if not routed:
        return AutoImageResourceStep(app_context=app_context, file_store=file_store, as_llm=model)
    app_context.registry = R
    return AutoResourceStep(
        app_context=app_context,
        dispatch_steps=[
            {"backend": "auto_image_resource_step", "file_store": file_store, "as_llm": model},
            {
                "backend": "auto_text_resource_step",
                "file_store": file_store,
                "agent_wrapper": FakeAgentWrapper(),
            },
        ],
    )


@dataclass
class AutoResourceTestEnv:
    """Started, isolated workspace shared by one test invocation."""

    workspace: Path
    app_context: object
    file_store: LocalFileStore

    def write_binary(self, relative_path: str, data: bytes) -> Path:
        """Write bytes relative to this workspace."""
        return write_binary(self.workspace / relative_path, data)

    def write_note(self, relative_path: str, source_resource: str, body: str = "old caption") -> Path:
        """Write a source-owned note relative to this workspace."""
        return write_note(self.workspace / relative_path, source_resource, body)

    def processor(self, model, *, routed: bool = False):
        """Build the direct processor or unified router for this workspace."""
        return image_processor(self.app_context, self.file_store, model, routed=routed)

    async def run(self, step, changes, **context_kwargs):
        """Run one processor invocation with a fresh runtime context."""
        return await step(RuntimeContext(changes=changes, **context_kwargs))


@pytest_asyncio.fixture
async def auto_resource_env(tmp_path, monkeypatch):
    """Yield a started resource-test workspace and always close its file store."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    app_context = make_app_context(workspace)
    file_store = LocalFileStore(name="test_store", embedding_store="")
    await file_store.start()
    _install_file_jobs(app_context, file_store)
    try:
        yield AutoResourceTestEnv(workspace, app_context, file_store)
    finally:
        await file_store.close()
