"""Native AgentScope input checks and image-agent lifecycle regressions (no network)."""

# pylint: disable=protected-access

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import frontmatter
import pytest
import yaml
from agentscope.message import DataBlock, Msg, TextBlock

from reme.application import Application
from reme.components import ApplicationContext, R
from reme.components.agent_wrapper import AsAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
from reme.steps.evolve.auto_image_resource import AutoImageResourceStep
from reme.steps.evolve.auto_resource import AutoResourceStep
from reme.steps.evolve.auto_text_resource import AutoTextResourceStep

from .auto_resource_test_support import FakeAgentWrapper, FakeImageAgentWrapper, caption_fields, image_bytes

pytest_plugins = ("unit.auto_resource_test_plugin",)
pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
@pytest.mark.parametrize(
    "options",
    [{}, {"include_images": True}, {"include_images": False}, {"include_images": "false"}, {"include_images": None}],
)
async def test_text_agent_keeps_main_reply_arguments_and_wrapper_defaults(existing, options, auto_resource_env):
    """Sharing interpretation must not override the text wrapper's optional settings."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/notes.txt"
    source = env.write_binary(source_path, "中文文本".encode())
    if existing:
        env.write_note("daily/2026-01-01/notes.md", f"[[{source_path}]]")
    wrapper = FakeAgentWrapper()
    step = AutoTextResourceStep(app_context=env.app_context, file_store=env.file_store, agent_wrapper=wrapper)
    with patch.object(wrapper, "reply", new=AsyncMock(wraps=wrapper.reply)) as reply:
        response = await env.run(
            step,
            [{"change": "modified" if existing else "added", "path": str(source)}],
            **options,
        )
    assert response.success
    assert response.answer == "ok"
    assert isinstance(wrapper.inputs, str)
    assert "中文文本" in wrapper.inputs
    reply.assert_awaited_once()
    assert reply.call_args.kwargs == {
        "system_prompt": step.prompt_format("system_prompt"),
        "job_tools": ["read", "edit", "frontmatter_update", "write"] if existing else ["write"],
        "session_id": str(uuid.uuid5(uuid.NAMESPACE_URL, source_path)),
    }
    tools = step.update_tools if existing else step.create_tools
    tools.append("custom_note_tool")
    with patch.object(wrapper, "reply", new=AsyncMock(wraps=wrapper.reply)) as reply:
        await env.run(step, [{"change": "modified", "path": str(source)}], **options)
    assert reply.call_args.kwargs["job_tools"] == tools
    assert reply.call_args.kwargs["session_id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, source_path))


async def test_native_image_input_preserves_context_formatter_and_scoped_tools(auto_resource_env):
    """Exercise real Agent construction/observe/format and real file tools, but no model call."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/brown-coat.png", image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("coat", "Brown coat", "A brown coat."))
    response = await env.run(env.processor(wrapper), [{"change": "added", "path": str(source)}])
    assert response.success
    message, options = wrapper.calls[0]
    assert isinstance(message, Msg)
    assert sum(isinstance(block, DataBlock) for block in message.content) == 1
    assert options["job_tools"] == ["write"]
    assert options["builtin_tools"] == []
    assert options["skills"] == []
    assert options["toolkit"] is None
    assert options["output_schema"] is None
    assert options["resume"] is None
    assert options["session_id"] is None
    assert "scope_note_tools" not in options
    assert options["injected_job_kwargs"]["_allowed_paths"] == ["daily/2026-01-01/brown-coat.md"]

    native = AsAgentWrapper(
        app_context=ApplicationContext(workspace_dir=str(env.workspace)),
        as_llm="",
        session_retention_days=0,
        session_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "component-default-session")),
        resume=str(uuid.uuid5(uuid.NAMESPACE_URL, "component-default-resume")),
    )
    native.as_llm = wrapper.as_llm
    # Reuse actual file-job adapters; only model inference is intentionally absent.
    native.app_context.jobs = env.app_context.jobs
    merged_options = native._merged_kwargs(options)
    assert merged_options["session_id"] is merged_options["resume"] is None
    agent, forwarded = await native._build_agent(message, **merged_options)
    assert forwarded is message
    assert agent.model is wrapper.as_llm.model
    await agent.observe(forwarded)
    await agent._limit_context_images(agent.context_config)
    formatted = await agent.model.formatter.format(agent.state.context)
    sent_image = next(block for block in formatted[0]["content"] if block["type"] == "image_url")
    original_image = next(block for block in message.content if isinstance(block, DataBlock))
    assert sent_image["image_url"]["url"] == (
        f"data:{original_image.source.media_type};base64,{original_image.source.data}"
    )
    restored = Msg.model_validate_json(agent.state.context[0].model_dump_json())
    assert next(block for block in restored.content if isinstance(block, DataBlock)) == original_image
    new_agent, _ = await native._build_agent(message, **merged_options)
    assert agent.state.session_id != new_agent.state.session_id
    assert agent.state.session_id != native.kwargs["session_id"]
    assert new_agent.state.session_id != native.kwargs["session_id"]
    assert not new_agent.state.context
    assert not (env.workspace / "mem_session").exists()

    outside = env.workspace / "unrelated.md"
    outside.write_text("preserve", encoding="utf-8")
    tool = native._make_tool(
        env.app_context.jobs["write"],
        injected_job_kwargs=options["injected_job_kwargs"],
    )
    result = await tool.call(path="unrelated.md", content="must not overwrite")
    assert "error" in str(result.state).lower()
    assert outside.read_text(encoding="utf-8") == "preserve"


async def test_message_blocks_do_not_implicitly_enable_image_policies(auto_resource_env):
    """Input shape alone must not select image tool scoping or session policy."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/notes.txt"
    wrapper = FakeAgentWrapper()
    step = AutoTextResourceStep(app_context=env.app_context, file_store=env.file_store, agent_wrapper=wrapper)
    step.context = RuntimeContext()
    with patch.object(wrapper, "reply", new=AsyncMock(wraps=wrapper.reply)) as reply:
        await step._interpret_resource(
            source_path,
            "2026-01-01",
            "notes",
            True,
            "中文文本",
            input_blocks=[TextBlock(text="Additional context")],
        )
    assert isinstance(wrapper.inputs, Msg)
    assert reply.call_args.kwargs == {
        "system_prompt": step.prompt_format("system_prompt"),
        "job_tools": ["write"],
        "session_id": str(uuid.uuid5(uuid.NAMESPACE_URL, source_path)),
    }


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("change", ["added", "modified", "deleted"])
async def test_include_images_false_skips_every_event_without_read_or_mutation(routed, change, auto_resource_env):
    """The opt-out is a full image lifecycle opt-out, including existing-note deletion."""
    env = auto_resource_env
    source = env.write_binary("resource/photo.png", image_bytes())
    old_note = env.write_note("daily/2026-01-01/old.md", "[[resource/photo.png]]")
    before = old_note.read_bytes()
    wrapper = FakeImageAgentWrapper("must not run")
    step = env.processor(wrapper, routed=routed)
    with patch.object(AutoImageResourceStep, "_read_image", side_effect=AssertionError("must not read")):
        response = await env.run(step, [{"change": change, "path": str(source)}], include_images=False)
    result = response.metadata["results"][0]
    assert response.success
    assert result["metadata"]["action"] == "skipped"
    assert result["metadata"]["reason"] == "include_images=false"
    assert result["metadata"]["modified"] is False
    assert not wrapper.calls
    assert old_note.read_bytes() == before
    assert not (env.workspace / "daily/2026-01-01.md").exists()


@pytest.mark.parametrize("routed", [False, True])
async def test_image_opt_out_does_not_bypass_resource_path_validation(routed, auto_resource_env):
    """Disabled image processing still rejects malformed external source paths."""
    env = auto_resource_env
    wrapper = FakeImageAgentWrapper("must not run")
    response = await env.run(
        env.processor(wrapper, routed=routed),
        [{"change": "added", "path": "resource/../private.png"}],
        include_images=False,
    )
    assert not response.success
    assert response.metadata["results"][0]["metadata"]["action"] == "failed"
    assert not wrapper.calls


async def test_disabled_images_keep_mixed_router_results_and_text_processing(auto_resource_env):
    """Routing keeps image ownership instead of handing image bytes to the text fallback."""
    env = auto_resource_env
    image = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    text = env.write_binary("resource/2026-01-01/notes.txt", "中文文本".encode())
    wrapper = FakeImageAgentWrapper("must not run")
    wrapper.app_context = env.app_context
    text_wrapper = FakeAgentWrapper()
    env.app_context.registry = R
    hook_calls = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hook_calls.append(kwargs)}
    step = AutoResourceStep(
        app_context=env.app_context,
        file_store=env.file_store,
        dispatch_steps=[
            {"backend": "auto_image_resource_step", "agent_wrapper": wrapper},
            {"backend": "auto_text_resource_step", "agent_wrapper": text_wrapper},
        ],
    )
    response = await env.run(
        step,
        [
            {"change": "added", "path": str(image)},
            {"change": "added", "path": str(text)},
        ],
        include_images=False,
    )
    assert response.success
    assert len(response.metadata["results"]) == 2
    assert response.metadata["results"][0]["metadata"]["reason"] == "include_images=false"
    assert "中文文本" in text_wrapper.inputs
    assert not wrapper.calls
    assert response.metadata["modified"] is False
    assert not hook_calls


@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
async def test_image_agent_failure_before_write_is_not_retried(existing, auto_resource_env):
    """A failure before writing must not create, repair, index, or retry a note."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/photo.png"
    source = env.write_binary(source_path, image_bytes())
    note = env.workspace / "daily/2026-01-01/photo.md"
    if existing:
        env.write_note("daily/2026-01-01/photo.md", f"[[{source_path}]]")
    before = note.read_bytes() if existing else None
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    wrapper = FakeImageAgentWrapper(error=RuntimeError("agent failed before writing"))
    response = await env.run(
        env.processor(wrapper, routed=True),
        [{"change": "modified" if existing else "added", "path": str(source)}],
    )
    result = response.metadata["results"][0]["metadata"]
    assert not response.success
    assert len(wrapper.calls) == 1
    assert result["action"] == "failed"
    assert result["modified"] is False
    assert (note.read_bytes() if note.exists() else None) == before
    assert not hooks
    assert not (env.workspace / "daily/2026-01-01.md").exists()


async def _cancel_after(task, ready):
    """Cancel the real invocation at an observed side-effect boundary and always join it."""
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("suffix", ["png", "txt"], ids=["image", "text"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["error", "cancel"])
async def test_written_resource_is_finalized_after_reply_failure(suffix, existing, cancelled, auto_resource_env):
    """Both processors finalize real writes while preserving the reply failure or task cancellation."""
    env = auto_resource_env
    source_path = f"resource/2026-01-01/photo.{suffix}"
    source = env.write_binary(source_path, image_bytes() if suffix == "png" else "中文文本".encode())
    note_path = "daily/2026-01-01/existing-card.md" if existing else "daily/2026-01-01/new-caption.md"
    target = note_path if existing else "daily/2026-01-01/photo.md"
    before = None
    if existing:
        before = env.write_note(note_path, f"[[{source_path}]]").read_bytes()
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    wrapper = FakeImageAgentWrapper()
    wrapper.app_context = env.app_context
    env.app_context.registry = R
    step = AutoResourceStep(
        app_context=env.app_context,
        dispatch_steps=[
            {
                "backend": "auto_image_resource_step" if suffix == "png" else "auto_text_resource_step",
                "file_store": env.file_store,
                "agent_wrapper": wrapper,
            },
        ],
    )
    written = asyncio.Event()

    async def write_then_fail(_inputs, **kwargs):
        tool = wrapper._make_tool(
            env.app_context.jobs["write"],
            injected_job_kwargs=kwargs.get("injected_job_kwargs"),
        )
        await tool.call(
            path=target,
            name="new-caption",
            description="New description",
            content=f"![[{source_path}]]\n\n## Caption\n\nNew caption." if suffix == "png" else "New text.",
            metadata={"source_resource": f"[[{source_path}]]"},
        )
        if cancelled:
            written.set()
            await asyncio.Event().wait()
        raise RuntimeError("agent failed after writing")

    with patch.object(wrapper, "reply", new=AsyncMock(side_effect=write_then_fail)) as reply:
        invocation = env.run(step, [{"change": "modified" if existing else "added", "path": str(source)}])
        if cancelled:
            await _cancel_after(asyncio.create_task(invocation), written)
            response = step.context.response
            result = response.metadata
        else:
            response = await invocation
            result = response.metadata["results"][0]["metadata"]
            assert result["action"] == "failed"
            assert result["error"] == "agent failed after writing"
            assert len(hooks) == 1
        reply.assert_awaited_once()
    assert not response.success
    assert result["path"] == note_path
    assert response.metadata["modified"] is result["modified"] is True
    note = env.workspace / note_path
    assert note.read_bytes() != before
    post = frontmatter.load(note)
    assert post["source_resource"] == f"[[{source_path}]]"
    if suffix == "png":
        assert post["kind"] == "image" and post["media_type"] == "image/png"
    assert post["name"] == Path(note_path).stem
    assert post["description"] == "New description"
    assert post.content.endswith("New caption." if suffix == "png" else "New text.")
    assert not (env.workspace / "daily/2026-01-01/photo.md").exists()
    index = (env.workspace / "daily/2026-01-01.md").read_text(encoding="utf-8")
    assert note_path in index and "New description" in index


@pytest.mark.parametrize("failure", ["index", "caption"])
async def test_agent_error_survives_failed_post_write_finalization(failure, auto_resource_env, monkeypatch):
    """Neither an index error nor invalid output may replace the original reply error."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(
        caption_fields("photo", "New description", "New caption." if failure == "index" else ""),
    )
    wrapper.note_metadata = {"source_resource": "[[resource/2026-01-01/photo.png]]"}
    wrapper.after_write_error = RuntimeError("agent failed after writing")
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    if failure == "index":
        monkeypatch.setattr(
            "reme.steps.evolve.base_auto_resource.refresh_day_index",
            AsyncMock(side_effect=RuntimeError("index refresh failed")),
        )
    response = await env.run(env.processor(wrapper, routed=True), [{"change": "added", "path": str(source)}])
    result = response.metadata["results"][0]["metadata"]
    assert not response.success
    assert result["action"] == "failed"
    assert result["error"] == "agent failed after writing"
    assert "agent failed after writing" in response.answer
    assert response.metadata["modified"] is result["modified"] is True
    assert result["path"] == "daily/2026-01-01/photo.md"
    assert len(wrapper.calls) == len(hooks) == 1
    note = frontmatter.load(env.workspace / result["path"])
    assert note["kind"] == "image" and note["media_type"] == "image/png"
    assert (env.workspace / "daily/2026-01-01.md").exists() is (failure != "index")


async def test_unchanged_image_write_does_not_trigger_failure_recovery(auto_resource_env):
    """An identical rewrite is not grounds to repair an existing note or emit a hook."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/photo.png"
    source = env.write_binary(source_path, image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("photo", "Same description", "Same caption."))
    wrapper.note_metadata = {"source_resource": f"[[{source_path}]]"}
    note_path = "daily/2026-01-01/photo.md"
    await env.app_context.jobs["write"](
        path=note_path,
        name="photo",
        description="Same description",
        content=f"![[{source_path}]]\n\n## Caption\n\nSame caption.\n",
        metadata=wrapper.note_metadata,
    )
    note = env.workspace / note_path
    before = note.read_bytes()
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    wrapper.after_write_error = RuntimeError("agent failed after identical write")
    response = await env.run(env.processor(wrapper, routed=True), [{"change": "modified", "path": str(source)}])
    assert not response.success
    assert response.metadata["modified"] is False
    assert note.read_bytes() == before
    assert not hooks
    assert len(wrapper.calls) == 1
    assert not (env.workspace / "daily/2026-01-01.md").exists()


@pytest.mark.parametrize(
    ("body", "valid"),
    [
        ("![[{source}]]\n\n## Caption\n\n一件棕色外衣。", True),
        ('![[{source}]]\n\n## Caption\n\nScreenshot of JSON:\n```json\n{{"count": 2}}\n```', True),
        ("![[{source}]]\n\n## Caption\n\n42", True),
        ("## Caption\n\nA coat.", False),
        ("![[resource/other.png]]\n\n## Caption\n\nA coat.", False),
        ("![[{source}]]\n\nA coat.", False),
        ("![[{source}]]\n\n## Caption\n\n  ", False),
        ('![[{source}]]\n\n## Caption\n\n{{"caption": "A coat."}}', False),
        ('![[{source}]]\n\n## Caption\n\n["A coat."]', False),
        ('![[{source}]]\n\n## Caption\n\n```json\n{{"caption": "A coat."}}\n```', False),
    ],
    ids=["text", "json-ocr", "number", "no-embed", "wrong-embed", "no-heading", "empty", "object", "array", "fence"],
)
async def test_image_note_body_is_validated_without_rolling_back_the_write(body, valid, auto_resource_env):
    """Validate actual Markdown, retaining already-written files and reporting their side effects."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/photo.png"
    source = env.write_binary(source_path, image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("photo", "Photo", "ignored by body override"))
    wrapper.note_body = body.format(source=source_path)
    wrapper.note_metadata = {"source_resource": f"[[{source_path}]]"}
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    response = await env.run(env.processor(wrapper, routed=True), [{"change": "added", "path": str(source)}])
    result = response.metadata["results"][0]["metadata"]
    assert response.success is valid
    assert result["modified"] is True
    assert result["action"] == ("added" if valid else "failed")
    assert len(wrapper.calls) == len(hooks) == 1
    note = frontmatter.load(env.workspace / result["path"])
    assert note.content == wrapper.note_body.strip()
    assert note["source_resource"] == f"[[{source_path}]]"
    assert note["kind"] == "image"
    assert note["media_type"] == "image/png"
    assert (env.workspace / "daily/2026-01-01.md").exists()


@pytest.mark.parametrize(
    ("before", "after", "valid"),
    [
        ({}, {}, True),
        ({}, {"status": "done"}, False),
        ({"status": "queued"}, {"status": "queued"}, True),
        ({"status": "queued"}, {"status": "done"}, False),
        ({"status": "queued"}, {}, False),
        ({"status": None}, {}, False),
    ],
    ids=["absent", "added", "preserved", "changed", "removed", "null-removed"],
)
async def test_image_agent_must_preserve_downstream_status(before, after, valid, auto_resource_env):
    """Presence and value matter; an unchanged downstream status is not an agent violation."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/photo.png"
    source = env.write_binary(source_path, image_bytes())
    note_path = "daily/2026-01-01/photo.md"
    env.write_note(note_path, f"[[{source_path}]]")
    if before:
        await env.app_context.jobs["frontmatter_update"](path=note_path, metadata=before)
    wrapper = FakeImageAgentWrapper(caption_fields("photo", "Updated photo", "A brown coat."))
    wrapper.note_metadata = {"source_resource": f"[[{source_path}]]", **after}
    response = await env.run(env.processor(wrapper), [{"change": "modified", "path": str(source)}])
    result = response.metadata["results"][0]["metadata"]
    assert response.success is valid
    assert result["modified"] is True
    if not valid:
        assert "status" in result["error"].lower()
    post = frontmatter.load(env.workspace / note_path)
    assert ("status" in post) is ("status" in after)
    assert post.get("status") == after.get("status")
    assert (env.workspace / "daily/2026-01-01.md").exists()


async def test_image_rejects_zero_image_budget(auto_resource_env):
    """A zero image budget must not yield filename-only memory."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper()
    wrapper.kwargs["context_config"] = {"max_image_num": 0}
    step = AutoImageResourceStep(app_context=env.app_context, file_store=env.file_store, agent_wrapper=wrapper)
    response = await env.run(step, [{"change": "added", "path": str(source)}])
    assert not response.success
    assert "context_config.max_image_num" in response.metadata["results"][0]["metadata"]["error"]
    assert response.metadata["results"][0]["metadata"]["modified"] is False
    assert not wrapper.calls
    assert not (env.workspace / "daily/2026-01-01/photo.md").exists()


@pytest.mark.parametrize(
    ("backend", "suffix", "job_options", "call_options", "expected_action"),
    [
        ("agentscope", "png", {}, {}, "added"),
        ("agentscope", "png", {"include_images": False}, {}, "skipped"),
        ("agentscope", "png", {"include_images": True}, {"include_images": False}, "skipped"),
        ("agentscope", "png", {"include_images": False}, {"include_images": True}, "added"),
        ("claude_code", "png", {}, {}, "failed"),
        ("claude_code", "png", {}, {"include_images": False}, "skipped"),
        ("claude_code", "txt", {}, {"include_images": True}, None),
    ],
    ids=["default", "job-disables", "call-disables", "call-enables", "other-backend", "disabled", "text"],
)
async def test_configured_wrapper_and_job_image_options(
    backend,
    suffix,
    job_options,
    call_options,
    expected_action,
    tmp_path,
):
    """Use registry-built wrappers and real jobs; fake only the external agent reply."""
    root = Path(__file__).resolve().parents[2]
    defaults = yaml.safe_load((root / "reme/config/default.yaml").read_text(encoding="utf-8"))
    jobs = {
        name: defaults["jobs"][name] for name in ("auto_resource", "write", "move", "frontmatter_update", "daily_list")
    }
    jobs["auto_resource"].update(job_options)
    jobs["auto_resource"]["steps"][0]["agent_wrapper"] = "resource_agent"
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components={
            "agent_wrapper": {"resource_agent": {"backend": backend, "as_llm": ""}},
            "file_store": {"default": {"backend": "local", "embedding_store": ""}},
            "file_graph": {"default": {"backend": "local"}},
            "keyword_index": {"default": {"backend": "bm25"}},
            "tokenizer": {"default": {"backend": "regex"}},
        },
        jobs=jobs,
    )
    source = tmp_path / f"resource/2026-01-01/photo.{suffix}"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(image_bytes() if suffix == "png" else "中文文本".encode())
    wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["resource_agent"]
    assert wrapper.backend == backend
    assert type(wrapper) is app.context.registry.get(ComponentEnum.AGENT_WRAPPER, backend)
    fake = (
        FakeImageAgentWrapper(caption_fields("photo", "Photo", "An image.")) if suffix == "png" else FakeAgentWrapper()
    )
    fake.app_context = app.context
    await app.start()
    try:
        with patch.object(wrapper, "reply", new=AsyncMock(side_effect=fake.reply)) as reply:
            response = await app.run_job(
                "auto_resource",
                changes=[{"change": "added", "path": str(source)}],
                **call_options,
            )
    finally:
        await app.close()
    result = response.metadata["results"][0]["metadata"]
    assert response.success is (expected_action != "failed")
    assert result.get("action") == expected_action
    assert reply.await_count == int(expected_action == "added" or suffix == "txt")
    if suffix == "txt":
        assert response.answer == "ok"
        assert isinstance(reply.call_args.args[0], str)
        assert set(reply.call_args.kwargs) == {"system_prompt", "job_tools", "session_id"}
    elif expected_action == "added":
        note = frontmatter.load(tmp_path / result["path"])
        assert note["source_resource"] == f"[[resource/2026-01-01/photo.{suffix}]]"
    elif expected_action == "skipped":
        assert result["reason"] == "include_images=false"
    else:
        assert "AgentScope wrapper" in result["error"]


@pytest.mark.parametrize("owner", [None, "[[resource/other.png]]"], ids=["missing-owner", "foreign-owner"])
async def test_partial_agent_write_cannot_claim_an_unowned_note(owner, auto_resource_env):
    """Report failed writes without claiming a note whose source ownership is missing or different."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("foreign", "Foreign", "A foreign-owned note."))
    wrapper.note_metadata = {"source_resource": owner} if owner else {}
    wrapper.after_write_error = RuntimeError("agent failed after writing")
    step = env.processor(wrapper)
    response = await env.run(step, [{"change": "added", "path": str(source)}])
    result = response.metadata["results"][0]["metadata"]
    assert result["error"] == "agent failed after writing"
    assert not response.success
    assert result["modified"] is response.metadata["modified"] is True
    note = env.workspace / "daily/2026-01-01/photo.md"
    before = note.read_bytes()
    post = frontmatter.load(note)
    assert post.get("source_resource") == owner
    assert "kind" not in post and "media_type" not in post
    assert not (env.workspace / "daily/2026-01-01/foreign.md").exists()
    assert not (env.workspace / "daily/2026-01-01.md").exists()
    deleted = await env.run(step, [{"change": "deleted", "path": str(source)}])
    assert deleted.success
    assert deleted.metadata["results"][0]["metadata"]["reason"] == "resource_note_not_found"
    assert note.read_bytes() == before
