"""Auto Memory adapts image input without changing its text or source contracts."""

# pylint: disable=protected-access,missing-function-docstring

import base64
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agentscope.formatter import DashScopeChatFormatter, OpenAIChatFormatter
from agentscope.message import Base64Source, DataBlock, Msg, TextBlock, URLSource
import httpx
import pytest
import yaml

from reme.application import Application
from reme.components import R
from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper
from reme.components.file_store import LocalFileStore
from reme.components.job import BaseJob
from reme.components.tag_index import LocalTagIndex
from reme.schema import ApplicationConfig
from reme.steps.evolve.auto_memory import AutoMemoryStep

from .test_auto_tag import _TaggingWrapper, _write_note

_DAY = "2026-09-01"
_SESSION = "image-input"


def _image(source=None):
    # Source validation/decoding belongs to the formatter/provider, not this Step.
    return DataBlock(
        id="duplicate-image-id",
        source=source or Base64Source(media_type="image/png", data="not-decoded-by-ReMe"),
    )


def _message(message_id="first", *, images=True, timestamp=f"{_DAY}T10:00:00"):
    return Msg.model_validate(
        {
            "id": message_id,
            "name": "Alice",
            "role": "user",
            "created_at": timestamp,
            "content": [TextBlock(text="Remember this observation."), *([_image()] if images else [])],
            "metadata": {"user_owned": {"nested": ["keep", 7]}},
        },
    )


def _saved_line(message):
    """Independent oracle for main's unchanged source serialization."""
    content = [
        block
        for block in message.content
        if block.type != "tool_result"
        and not (block.type == "data" and getattr(block.source, "type", None) == "base64")
    ]
    return (message.model_copy(update={"content": content}).model_dump_json() + "\n").encode("utf-8")


@pytest.fixture(name="setup")
def memory_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper = AsAgentWrapper(backend="agentscope", as_llm="")
    wrapper.reply = AsyncMock(return_value={"result": "ok"})
    store = LocalFileStore(embedding_store="")
    app = SimpleNamespace(
        registry=R,
        metadata={},
        app_config=ApplicationConfig(workspace_dir=str(tmp_path)),
        jobs={},
        components={},
    )
    step = AutoMemoryStep(app_context=app, file_store=store, agent_wrapper=wrapper)
    monkeypatch.setattr(step, "_list_session_note", AsyncMock(return_value=None))
    return step, wrapper, tmp_path / "session" / "dialog" / f"{_SESSION}.jsonl"


async def _run(step, messages, **kwargs):
    await step(session_id=_SESSION, date=_DAY, messages=messages, **kwargs)
    return step.context.response


def test_only_include_images_is_exposed_and_disabled_by_default():
    path = Path(__file__).resolve().parents[2] / "reme/config/default.yaml"
    job = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["auto_memory"]
    assert job["parameters"]["properties"]["include_images"]["default"] is False
    assert {"supports_vision", "image_mode"}.isdisjoint(job["parameters"]["properties"])
    assert job["steps"] == [{"backend": "auto_memory_step"}, {"backend": "auto_tag_step"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,images",
    [
        ({}, True),
        ({"include_images": False}, True),
        ({"include_images": False}, False),
        ({"include_images": True}, False),
        ({"include_images": "false"}, False),
        ({"include_images": 0}, False),
        ({"include_images": None}, False),
    ],
)
async def test_text_path_keeps_main_input_kwargs_metadata_and_jsonl(setup, monkeypatch, options, images):
    step, _, path = setup
    wrapper = CcAgentWrapper(backend="claude_code")
    wrapper.reply = AsyncMock(return_value={"result": "ok"})
    wrapper.kwargs["context_config"] = {"max_image_num": 0}
    step.kwargs["agent_wrapper"] = wrapper
    message = _message(images=images)
    message.content.append(DataBlock(source=URLSource(media_type="application/pdf", url="file:///not-read.pdf")))
    before = message.model_dump()
    extra = {"model_config": {"max_retries": 2}}
    expected = step.prompt_format(
        "user_message_create",
        today=_DAY,
        note="(none)",
        note_path="",
        session_id=_SESSION,
        session_file=f"session/dialog/{_SESSION}.jsonl",
        history=step._format_history([message]),
    )
    events = []
    save, history = step._save_session_messages, step._format_history

    async def save_source(*args):
        await save(*args)
        events.append("save")

    def format_history(messages):
        events.append("history")
        return history(messages)

    def reply_kwargs(_day):
        events.append("reply_kwargs")
        return extra

    monkeypatch.setattr(step, "_save_session_messages", save_source)
    monkeypatch.setattr(step, "_format_history", format_history)
    monkeypatch.setattr(step, "_reply_extra_kwargs", reply_kwargs)

    response = await _run(step, [message], **options)

    wrapper.reply.assert_awaited_once_with(
        expected,
        system_prompt=step.prompt_format("system_prompt"),
        job_tools=["daily_write"],
        **extra,
    )
    assert isinstance(wrapper.reply.call_args.args[0], str)
    assert response.success is True and response.answer == "ok"
    assert response.metadata == {"date": _DAY, "path": None, "created": False, "modified": False, "n_messages": 1}
    assert path.read_bytes() == _saved_line(message)
    assert message.model_dump() == before
    assert wrapper.kwargs["context_config"] == {"max_image_num": 0}
    assert events == ["save", "history", "reply_kwargs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("formatter_type", [OpenAIChatFormatter, DashScopeChatFormatter])
@pytest.mark.parametrize("url", ["https://images.example.org/x.png?version=2", "http://images.example.org/x.png"])
async def test_sources_interleave_unchanged_through_native_formatters(setup, monkeypatch, formatter_type, url):
    step, wrapper, path = setup
    message = _message()
    message.content[0].text += " Keep literal [Image 1]."
    message.content.extend([TextBlock(text="Between images."), _image(URLSource(media_type="image/png", url=url))])
    message.content.append(TextBlock(text="After both images."))
    original = copy.deepcopy(message.model_dump())
    forbidden = Mock(side_effect=AssertionError("Auto Memory must not download or decode image sources"))
    monkeypatch.setattr(base64, "b64decode", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)

    response = await _run(step, [message], include_images=True)

    inputs, options = wrapper.reply.call_args.args[0], wrapper.reply.call_args.kwargs
    assert isinstance(inputs, Msg) and inputs.role == "user"
    assert [block.type for block in inputs.content] == ["text", "data", "text", "data", "text"]
    assert f"[Alice @ {message.created_at}]" in inputs.content[0].text
    assert "Remember this observation." in inputs.content[0].text
    assert inputs.get_text_content().count("[Image 1]") == 1
    assert "__reme_image_" not in inputs.get_text_content()
    assert inputs.content[2].text.strip() == "Between images."
    assert inputs.content[-1].text.index("After both images.") < inputs.content[-1].text.index("# Your Task")
    assert [inputs.content[index].model_dump() for index in (1, 3)] == [
        message.content[index].model_dump() for index in (1, 3)
    ]
    assert options == {"system_prompt": step.prompt_format("system_prompt"), "job_tools": ["daily_write"]}
    assert "auto_memory_images" not in response.metadata
    assert path.read_bytes() == _saved_line(message)
    assert message.model_dump() == original
    formatted = await formatter_type().format([inputs])
    parts = formatted[0]["content"]
    assert [part["type"] for part in parts] == ["text", "image_url", "text", "image_url", "text"]
    assert [parts[index]["image_url"]["url"] for index in (1, 3)] == [
        f"data:image/png;base64,{message.content[1].source.data}",
        url,
    ]
    forbidden.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["backend", "file", "ftp", "default-limit"])
async def test_static_errors_precede_any_source_session_write(setup, existing, failure):
    step, wrapper, path = setup
    old = _message("old", images=False, timestamp=f"{_DAY}T09:00:00")
    before = _saved_line(old)
    if existing:
        path.parent.mkdir(parents=True)
        path.write_bytes(before)
    message, options = _message(), {"include_images": True}
    error, match = ValueError, "max_image_num"
    if failure == "backend":
        step.kwargs["agent_wrapper"] = CcAgentWrapper(backend="claude_code")
        error, match = NotImplementedError, "AgentScope"
    elif failure in ("file", "ftp"):
        message.content[1].source = URLSource(media_type="image/png", url=f"{failure}:///private/image.png")
        match = "Base64|base64|HTTP|http"
    else:
        message.content = [_image() for _ in range(6)]

    with pytest.raises(error, match=match):
        await _run(step, [message], **options)

    assert path.read_bytes() == before if existing else not path.exists()
    wrapper.reply.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 1, [], "false"])
async def test_empty_messages_ignore_image_option_and_keep_main_skip_behavior(setup, value):
    step, wrapper, path = setup
    response = await _run(step, [], include_images=value)
    assert response.success is True and response.answer == "Skipped: no messages"
    assert response.metadata == {"date": _DAY, "modified": False, "n_messages": 0}
    assert not path.exists()
    wrapper.reply.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 5, 6])
@pytest.mark.parametrize("override", [False, True])
async def test_effective_image_limit_is_checked_without_modifying_configs(setup, monkeypatch, limit, override):
    step, wrapper, path = setup
    component = {"max_image_num": 9, "trigger_ratio": 0.7} if override else {"max_image_num": limit}
    wrapper.kwargs["context_config"] = component
    extra = {"context_config": {"max_image_num": limit}} if override else {}
    snapshots = copy.deepcopy((component, extra))
    monkeypatch.setattr(step, "_reply_extra_kwargs", Mock(return_value=extra))
    message = _message()
    message.content = [_image() for _ in range(6)]

    if limit < 6:
        with pytest.raises(ValueError, match="max_image_num"):
            await _run(step, [message], include_images=True)
        assert not path.exists()
        wrapper.reply.assert_not_called()
    else:
        await _run(step, [message], include_images=True)
        assert isinstance(wrapper.reply.call_args.args[0], Msg)
        assert wrapper.reply.call_args.kwargs == {
            "system_prompt": step.prompt_format("system_prompt"),
            "job_tools": ["daily_write"],
            **extra,
        }
    assert (component, extra) == snapshots


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [{}, None])
async def test_empty_context_override_uses_sdk_default_not_component_limit(setup, monkeypatch, override):
    step, wrapper, path = setup
    wrapper.kwargs["context_config"] = {"max_image_num": 10}
    monkeypatch.setattr(step, "_reply_extra_kwargs", Mock(return_value={"context_config": override}))
    message = _message()
    message.content = [_image() for _ in range(6)]
    with pytest.raises(ValueError, match="max_image_num"):
        await _run(step, [message], include_images=True)
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("language,existing", [("en", False), ("zh", True)])
async def test_full_history_hook_image_only_turn_and_update_boundaries(setup, monkeypatch, language, existing):
    step, wrapper, _ = setup
    step.prompt.language = language
    note_path = f"daily/{_DAY}/existing.md"
    if existing:
        monkeypatch.setattr(step, "_list_session_note", AsyncMock(return_value={"path": note_path}))
        monkeypatch.setattr(step, "_ensure_session_frontmatter", AsyncMock())
        monkeypatch.setattr(step, "_rename_from_frontmatter_name", AsyncMock(return_value=note_path))
        monkeypatch.setattr("reme.steps.evolve.auto_memory.refresh_day_index", AsyncMock(return_value={}))
    first, second = _message(), _message("second", timestamp=f"{_DAY}T11:00:00")
    second.name, second.content = "Bob", [_image()]
    originals = [message.model_dump() for message in (first, second)]
    hook = Mock(
        side_effect=lambda messages: "Source excerpt L1-L2\n"
        + "\n".join(
            f"[L{index} {message.name} @ {message.created_at}]\n{message.get_text_content()}"
            for index, message in enumerate(messages, 1)
        ),
    )
    monkeypatch.setattr(step, "_format_history", hook)

    await _run(step, [first, second], include_images=True)

    inputs, options = wrapper.reply.call_args.args[0], wrapper.reply.call_args.kwargs
    hook.assert_called_once()
    assert len(hook.call_args.args[0]) == 2
    assert inputs.get_text_content().count("Source excerpt L1-L2") == 1
    assert [block.type for block in inputs.content] == ["text", "data", "text", "data", "text"]
    assert "Alice @" in inputs.content[0].text and "Bob @" in inputs.content[2].text
    assert ("# Your Task" if language == "en" else "# 你的任务") in inputs.content[-1].text
    assert options["job_tools"] == (step.update_tools if existing else step.create_tools)
    if existing:
        assert note_path in inputs.content[0].text
        assert options["injected_job_kwargs"] == {"_allowed_paths": [note_path]}
    assert [message.model_dump() for message in (first, second)] == originals


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_provider_error_is_not_retried_and_keeps_main_saved_source(setup, enabled):
    step, wrapper, path = setup
    error = RuntimeError("provider rejected this request")
    wrapper.reply.side_effect = error
    message = _message()
    with pytest.raises(RuntimeError) as raised:
        await _run(step, [message], include_images=enabled)
    assert raised.value is error
    wrapper.reply.assert_awaited_once()
    assert isinstance(wrapper.reply.call_args.args[0], Msg if enabled else str)
    assert path.read_bytes() == _saved_line(message)
    assert "auto_memory_images" not in step.context.response.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend,configured,options,images,expected",
    [
        ("agentscope", True, {}, True, Msg),
        ("agentscope", False, {}, True, str),
        ("agentscope", True, {"include_images": False}, True, str),
        ("agentscope", False, {"include_images": True}, True, Msg),
        ("claude_code", True, {}, True, "AgentScope"),
        ("claude_code", False, {}, True, str),
        ("claude_code", True, {}, False, str),
        ("claude_code", False, {"include_images": "true"}, False, str),
    ],
)
async def test_configured_backend_and_job_switch(tmp_path, monkeypatch, backend, configured, options, images, expected):
    config = Path(__file__).resolve().parents[2] / "reme/config/default.yaml"
    job_config = yaml.safe_load(config.read_text(encoding="utf-8"))["jobs"]["auto_memory"]
    job_config.update(include_images=configured, steps=[{"backend": "auto_memory_step"}])
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components={
            "agent_wrapper": {"default": {"backend": backend, "as_llm": ""}},
            "file_store": {"default": {"backend": "local", "embedding_store": ""}},
        },
        jobs={"auto_memory": job_config},
    )
    wrapper = app.context.components["agent_wrapper"]["default"]
    assert isinstance(wrapper, AsAgentWrapper if backend == "agentscope" else CcAgentWrapper)
    assert wrapper.backend == backend
    wrapper.reply = AsyncMock(return_value={"result": "ok"})
    monkeypatch.setattr(AutoMemoryStep, "_list_session_note", AsyncMock(return_value=None))
    job = app.context.jobs["auto_memory"]
    await job.start()
    try:
        response = await job(session_id=_SESSION, date=_DAY, messages=[_message(images=images)], **options)
    finally:
        await job.close()
    if isinstance(expected, str):
        assert response.success is False and expected in response.answer
        assert not (tmp_path / "session/dialog" / f"{_SESSION}.jsonl").exists()
        wrapper.reply.assert_not_called()
    else:
        assert response.success is True
        assert isinstance(wrapper.reply.call_args.args[0], expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("history", ["append", "backfill", "same-id", "replace-and-backfill"])
async def test_history_merge_keeps_main_source_contract(setup, enabled, history):
    step, _, path = setup
    original = _message()
    await _run(step, [original], include_images=enabled)
    initial = path.read_bytes()
    replacement = original.model_copy(deep=True)
    replacement.content[0].text = "Changed same-ID text."
    older = _message("older", images=False, timestamp=f"{_DAY}T09:00:00")
    later = _message("later", images=False, timestamp=f"{_DAY}T11:00:00")
    if history == "append":
        messages, expected = [original, later], initial + _saved_line(later)
    elif history == "backfill":
        messages, expected = [older, original], _saved_line(older) + initial
    elif history == "same-id":
        messages, expected = [replacement], initial
    else:
        messages, expected = [older, replacement], _saved_line(older) + _saved_line(replacement)
    before = [message.model_dump() for message in messages]

    await _run(step, messages, include_images=enabled)

    assert path.read_bytes() == expected
    assert [message.model_dump() for message in messages] == before


@pytest.mark.asyncio
async def test_default_job_still_passes_memory_changes_to_auto_tag(setup, monkeypatch):
    step, wrapper, session_path = setup
    workspace = session_path.parents[2]
    note_path = f"daily/{_DAY}/image-memory.md"
    target = workspace / note_path
    step.file_store.tag_index = LocalTagIndex(max_tags_per_file=3)
    tagger = _TaggingWrapper(workspace, tags=["OpenAI"])

    async def find_note(*_args):
        return {"path": note_path} if target.exists() else None

    async def write_memory(*_args, **_kwargs):
        _write_note(target)
        return {"result": "Memory written."}

    monkeypatch.setattr(AutoMemoryStep, "_list_session_note", find_note)
    wrapper.reply.side_effect = write_memory
    config = Path(__file__).resolve().parents[2] / "reme/config/default.yaml"
    steps = yaml.safe_load(config.read_text(encoding="utf-8"))["jobs"]["auto_memory"]["steps"]
    for definition, agent in zip(steps, (wrapper, tagger)):
        definition.update(file_store=step.file_store, agent_wrapper=agent)
    job = BaseJob(app_context=step.app_context, steps=steps)
    await job.start()
    try:
        response = await job(session_id=_SESSION, date=_DAY, messages=[_message()], include_images=True)
    finally:
        await job.close()
    assert response.success is True and response.answer == "Memory written."
    assert response.metadata["auto_tag"]["processed"] == response.metadata["auto_tag"]["succeeded"] == 1
    assert response.metadata["auto_tag"]["results"][0]["path"] == response.metadata["path"] == note_path
    assert "auto_memory_images" not in response.metadata
    assert isinstance(wrapper.reply.call_args.args[0], Msg)
    assert len(tagger.calls) == 1
