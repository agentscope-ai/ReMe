"""Shared test harness for auto-resource processor and router tests."""

import io
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest_asyncio
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from PIL import Image

from reme.components import R
from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper
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


class FakeImageAgentWrapper(AsAgentWrapper):
    """Fake only the agent reply; write through the actual scoped ReMe job tool."""

    def __init__(
        self,
        content: dict | str = "A resource image.",
        *,
        error: Exception | None = None,
        perform_write: bool = True,
    ):
        super().__init__(backend="agentscope", as_llm="", session_retention_days=0)
        self.content = content
        self.error = error
        self.perform_write = perform_write
        self.calls: list[tuple[Msg, dict]] = []
        self.after_write_error: BaseException | None = None
        self.note_metadata: dict = {}
        self.note_body: str | None = None
        self.as_llm = SimpleNamespace(model=SimpleNamespace(formatter=OpenAIChatFormatter()))

    async def reply(self, inputs, **kwargs) -> dict:
        """Emulate a tool-writing agent, never a schema or provider response."""
        self.calls.append((inputs, kwargs))
        if self.error is not None:
            raise self.error
        if not self.perform_write:
            return {"result": str(self.content)}
        assert isinstance(inputs, Msg)
        assert kwargs.get("output_schema") is None
        target = kwargs["injected_job_kwargs"]["_allowed_paths"]
        assert len(target) == 1
        assert "write" in kwargs["job_tools"]
        prompt = inputs.get_text_content()
        source = re.search(r"resource/[^\s\]\n]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|heic)", prompt, re.IGNORECASE)
        assert source is not None, prompt
        fields = self.content if isinstance(self.content, dict) else {"caption": self.content}
        caption = fields.get("caption", "")
        content = (
            self.note_body if self.note_body is not None else f"![[{source.group()}]]\n\n## Caption\n\n{caption}\n"
        )
        tool = self._make_tool(
            self.app_context.jobs["write"],
            injected_job_kwargs=kwargs["injected_job_kwargs"],
        )
        response = await tool.call(
            path=target[0],
            name=fields.get("name") or Path(target[0]).stem,
            description=fields.get("description") or str(caption)[:120],
            content=content,
            metadata=self.note_metadata,
        )
        if self.after_write_error is not None:
            raise self.after_write_error
        return {"result": "Saved image note", "tool_result": response}


class FlakyImageAgentWrapper(FakeImageAgentWrapper):
    """Fail one agent reply, then use the real scoped write job."""

    async def reply(self, inputs, **kwargs) -> dict:
        """Fail once without writing, then recover for the next resource."""
        if not self.calls:
            self.calls.append((inputs, kwargs))
            raise RuntimeError("image agent unavailable")
        return await super().reply(inputs, **kwargs)


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
        self.name = "write"
        self.description = "Write an isolated test note"
        self.parameters = {
            "type": "object",
            "properties": {
                **{key: {"type": "string"} for key in ("path", "name", "description", "content")},
                "metadata": {"type": "object"},
            },
            "required": ["path", "content"],
        }

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


def caption_fields(name: str, description: str, caption: str) -> dict:
    """Build the fake agent's intended note content; not a model JSON response."""
    return {"name": name, "description": description, "caption": caption}


def image_processor(app_context, file_store, model, *, routed: bool, **kwargs):
    """Build either the image processor or the public unified-router path."""
    model.app_context = app_context
    if not routed:
        return AutoImageResourceStep(app_context=app_context, file_store=file_store, agent_wrapper=model, **kwargs)
    app_context.registry = R
    return AutoResourceStep(
        app_context=app_context,
        **kwargs,
        dispatch_steps=[
            {"backend": "auto_image_resource_step", "file_store": file_store, "agent_wrapper": model},
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

    def processor(self, model, *, routed: bool = False, **kwargs):
        """Build the direct processor or unified router for this workspace."""
        return image_processor(self.app_context, self.file_store, model, routed=routed, **kwargs)

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
