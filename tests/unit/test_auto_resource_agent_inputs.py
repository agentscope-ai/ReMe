"""Native AgentScope input checks and image-agent lifecycle regressions (no network)."""

# pylint: disable=protected-access

import asyncio
import os
from unittest.mock import patch

import frontmatter
import pytest
from agentscope.message import DataBlock, Msg

from reme.components import ApplicationContext, R
from reme.components.agent_wrapper import AsAgentWrapper
from reme.components.job import BaseJob
from reme.steps.evolve.auto_image_resource import AutoImageResourceStep
from reme.steps.evolve.auto_resource import AutoResourceStep

from .auto_resource_test_support import FakeAgentWrapper, FakeImageAgentWrapper, caption_fields, image_bytes

pytest_plugins = ("unit.auto_resource_test_plugin",)
pytestmark = pytest.mark.asyncio


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
    assert options["injected_job_kwargs"]["_allowed_paths"] == ["daily/2026-01-01/brown-coat.md"]

    native = AsAgentWrapper(
        app_context=ApplicationContext(workspace_dir=str(env.workspace)),
        as_llm="",
        session_retention_days=0,
    )
    native.as_llm = wrapper.as_llm
    # Reuse actual file-job adapters; only model inference is intentionally absent.
    native.app_context.jobs = env.app_context.jobs
    agent, forwarded = await native._build_agent(message, **options)
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
    new_agent, _ = await native._build_agent(message, **options)
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


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("change", ["added", "modified", "deleted"])
async def test_include_images_false_skips_every_event_without_read_or_mutation(routed, change, auto_resource_env):
    """The opt-out is a full image lifecycle opt-out, including existing-note deletion."""
    env = auto_resource_env
    source = env.write_binary("resource/photo.png", image_bytes())
    old_note = env.write_note("daily/2026-01-01/old.md", "[[resource/photo.png]]")
    before = old_note.read_bytes()
    wrapper = FakeImageAgentWrapper("must not run")
    step = env.processor(wrapper, routed=routed, include_images=False)
    with patch.object(AutoImageResourceStep, "_read_image", side_effect=AssertionError("must not read")):
        response = await env.run(step, [{"change": change, "path": str(source)}])
    result = response.metadata["results"][0]
    assert response.success
    assert result["metadata"]["action"] == "skipped"
    assert result["metadata"]["reason"] == "include_images=false"
    assert result["metadata"]["modified"] is False
    assert not wrapper.calls
    assert old_note.read_bytes() == before
    assert not (env.workspace / "daily/2026-01-01.md").exists()


@pytest.mark.parametrize("value", ["false", 0, None])
async def test_include_images_requires_a_real_boolean(value, auto_resource_env):
    """Configuration values must not accidentally enable image interpretation by truthiness."""
    env = auto_resource_env
    source = env.write_binary("resource/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper("must not run")
    response = await env.run(
        env.processor(wrapper, include_images=value),
        [{"change": "added", "path": str(source)}],
    )
    assert not response.success
    assert "include_images must be a boolean" in response.metadata["results"][0]["metadata"]["error"]
    assert not wrapper.calls


@pytest.mark.parametrize("routed", [False, True])
async def test_image_opt_out_does_not_bypass_resource_path_validation(routed, auto_resource_env):
    """Disabled image processing still rejects malformed external source paths."""
    env = auto_resource_env
    wrapper = FakeImageAgentWrapper("must not run")
    response = await env.run(
        env.processor(wrapper, routed=routed, include_images=False),
        [{"change": "added", "path": "resource/../private.png"}],
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
        include_images=False,
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
    )
    assert response.success
    assert len(response.metadata["results"]) == 2
    assert response.metadata["results"][0]["metadata"]["reason"] == "include_images=false"
    assert "中文文本" in text_wrapper.inputs
    assert not wrapper.calls
    assert response.metadata["modified"] is False
    assert not hook_calls


async def test_image_agent_failure_before_write_is_not_retried(auto_resource_env):
    """A failed agent call must not create a note or trigger an implicit retry."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(error=RuntimeError("agent failed before writing"))
    response = await env.run(env.processor(wrapper), [{"change": "added", "path": str(source)}])
    result = response.metadata["results"][0]["metadata"]
    assert not response.success
    assert len(wrapper.calls) == 1
    assert result["action"] == "failed"
    assert result["modified"] is False
    assert not (env.workspace / "daily/2026-01-01/photo.md").exists()


@pytest.mark.parametrize("body", ["", "   "])
async def test_image_agent_empty_caption_write_is_a_reportable_failure(body, auto_resource_env):
    """An embed/header alone does not constitute a caption; already-written bytes remain observable."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("photo", "", body))
    response = await env.run(env.processor(wrapper), [{"change": "added", "path": str(source)}])
    result = response.metadata["results"][0]["metadata"]
    assert not response.success
    assert result["modified"] is True
    assert result["action"] == "failed"


@pytest.mark.parametrize("invalid_config", ["non-agentscope-wrapper", "zero-image-budget"])
async def test_image_rejects_text_only_agent_configuration(invalid_config, auto_resource_env):
    """Neither an unsupported wrapper nor a zero image budget may yield filename-only memory."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeAgentWrapper() if invalid_config == "non-agentscope-wrapper" else FakeImageAgentWrapper()
    expected_error = "AgentScope wrapper"
    if invalid_config == "zero-image-budget":
        wrapper.kwargs["context_config"] = {"max_image_num": 0}
        expected_error = "context_config.max_image_num"
    step = AutoImageResourceStep(app_context=env.app_context, file_store=env.file_store, agent_wrapper=wrapper)
    response = await env.run(step, [{"change": "added", "path": str(source)}])
    assert not response.success
    assert expected_error in response.metadata["results"][0]["metadata"]["error"]
    assert response.metadata["results"][0]["metadata"]["modified"] is False
    assert not (wrapper.inputs if isinstance(wrapper, FakeAgentWrapper) else wrapper.calls)
    assert not (env.workspace / "daily/2026-01-01/photo.md").exists()


@pytest.mark.parametrize(
    ("job_value", "step_value", "call_value", "expected_enabled"),
    [
        (None, None, None, True),
        (None, False, None, False),
        (False, True, None, False),
        (True, False, None, True),
        (True, True, False, False),
        (False, False, True, True),
    ],
    ids=["default", "step", "job-disables", "job-enables", "call-disables", "call-enables"],
)
async def test_include_images_job_and_call_precedence(
    job_value,
    step_value,
    call_value,
    expected_enabled,
    auto_resource_env,
):
    """A real public Job resolves call context > Job defaults > image Step > true."""
    env = auto_resource_env
    env.app_context.registry = R.copy()
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("photo", "Photo", "An image."))
    wrapper.app_context = env.app_context
    image_spec = {"backend": "auto_image_resource_step", "agent_wrapper": wrapper}
    if step_value is not None:
        image_spec["include_images"] = step_value
    job_options = {} if job_value is None else {"include_images": job_value}
    job = BaseJob(
        name="auto_resource",
        app_context=env.app_context,
        steps=[
            {
                "backend": "auto_resource_step",
                "file_store": env.file_store,
                "dispatch_steps": [image_spec, "auto_text_resource_step"],
            },
        ],
        **job_options,
    )
    await job.start()
    try:
        call_options = {} if call_value is None else {"include_images": call_value}
        response = await job(changes=[{"change": "added", "path": str(source)}], **call_options)
    finally:
        await job.close()
    assert response.success
    assert bool(wrapper.calls) is expected_enabled
    assert response.metadata["modified"] is expected_enabled
    if not expected_enabled:
        assert response.metadata["results"][0]["metadata"]["reason"] == "include_images=false"


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("source_format", ["PNG", "JPEG"], ids=["same-mime", "changed-source-mime"])
async def test_empty_agent_write_restores_existing_valid_image_note(routed, source_format, auto_resource_env):
    """A blank agent-generated replacement must not destroy a prior valid caption."""
    env = auto_resource_env
    source = env.write_binary("resource/photo.png", image_bytes(source_format))
    note = env.write_note("daily/2026-01-01/original.md", "[[resource/photo.png]]", "Original visual facts.")
    before = note.read_bytes()
    wrapper = FakeImageAgentWrapper(caption_fields("empty-replacement", "", "   "))
    response = await env.run(env.processor(wrapper, routed=routed), [{"change": "modified", "path": str(source)}])
    assert not response.success
    assert len(wrapper.calls) == 1
    assert note.read_bytes() == before
    assert response.metadata["results"][0]["metadata"]["modified"] is False
    assert not (env.workspace / "daily/2026-01-01/empty-replacement.md").exists()


async def test_invalid_image_note_restore_preserves_intervening_editor_changes(auto_resource_env):
    """An optimistic save conflict cannot force an old snapshot over a newer user edit."""
    env = auto_resource_env
    source = env.write_binary("resource/photo.png", image_bytes())
    note = env.write_note("daily/2026-01-01/original.md", "[[resource/photo.png]]", "Original visual facts.")
    real_save = env.app_context.jobs["save"]
    intervening_content = b"A concurrent editor update that must survive.\n"
    save_results = []

    async def edit_before_restore(**kwargs):
        assert kwargs["path"] == "daily/2026-01-01/original.md"
        assert kwargs["_allowed_paths"] == [kwargs["path"]]
        assert kwargs["expected_mtime"]
        modified_stat = note.stat()
        note.write_bytes(intervening_content)
        os.utime(note, (modified_stat.st_atime, modified_stat.st_mtime + 2))
        result = await real_save(**kwargs)
        save_results.append(result)
        return result

    env.app_context.jobs["save"] = edit_before_restore
    wrapper = FakeImageAgentWrapper(caption_fields("empty", "", "   "))
    response = await env.run(env.processor(wrapper), [{"change": "modified", "path": str(source)}])
    assert len(save_results) == 1
    assert not save_results[0].success
    assert save_results[0].metadata["code"] == "file_conflict"
    assert not response.success
    assert response.metadata["results"][0]["metadata"]["modified"] is True
    assert note.read_bytes() == intervening_content


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("retry_first", [False, True], ids=["delete-partial-note", "retry-then-delete"])
@pytest.mark.parametrize("failure", ["agent-error", "empty-caption"])
async def test_partial_agent_write_remains_owned_for_retry_and_delete(routed, retry_first, failure, auto_resource_env):
    """Agent and content-validation failures retain ownership for subsequent retry/delete."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes("JPEG"))
    caption = "Visible photo content." if failure == "agent-error" else ""
    wrapper = FakeImageAgentWrapper(caption_fields("suggested-name", "Photo", caption))
    if failure == "agent-error":
        wrapper.after_write_error = RuntimeError("agent failed after writing")
    step = env.processor(wrapper, routed=routed)
    response = await env.run(step, [{"change": "added", "path": str(source)}])
    assert not response.success
    assert len(wrapper.calls) == 1
    result = response.metadata["results"][0]["metadata"]
    assert result["action"] == "failed"
    assert result["modified"] is True
    if failure == "empty-caption":
        assert "no usable content" in result["error"]
    note_path = env.workspace / "daily/2026-01-01/photo.md"
    post = frontmatter.load(note_path)
    assert post["source_resource"] == "[[resource/2026-01-01/photo.png]]"
    assert post["kind"] == "image"
    assert post["media_type"] == "image/jpeg"

    if retry_first:
        wrapper.after_write_error = None
        wrapper.content = caption_fields("another-name", "Updated photo", "Updated visual facts.")
        retried = await env.run(step, [{"change": "modified", "path": str(source)}])
        assert retried.success
        metadata = retried.metadata["results"][0]["metadata"]
        assert metadata["created"] is False
        assert metadata["path"] == "daily/2026-01-01/photo.md"
        assert "Updated visual facts." in note_path.read_text(encoding="utf-8")
        assert list(note_path.parent.glob("*.md")) == [note_path]

    source.unlink()
    deleted = await env.run(step, [{"change": "deleted", "path": str(source)}])
    assert deleted.success
    assert deleted.metadata["results"][0]["metadata"]["action"] == "deleted"
    assert not note_path.exists()
    assert len(wrapper.calls) == (2 if retry_first else 1)


async def test_partial_agent_write_cannot_claim_an_explicit_foreign_owner(auto_resource_env):
    """Best-effort failure cleanup must not retag another resource's explicit ownership."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("foreign", "Foreign", "A foreign-owned note."))
    wrapper.note_metadata = {"source_resource": "[[resource/other.png]]"}
    wrapper.after_write_error = RuntimeError("agent failed after writing")
    step = env.processor(wrapper)
    response = await env.run(step, [{"change": "added", "path": str(source)}])
    assert not response.success
    note = env.workspace / "daily/2026-01-01/photo.md"
    before = note.read_bytes()
    assert frontmatter.load(note)["source_resource"] == "[[resource/other.png]]"
    deleted = await env.run(step, [{"change": "deleted", "path": str(source)}])
    assert deleted.success
    assert deleted.metadata["results"][0]["metadata"]["reason"] == "resource_note_not_found"
    assert note.read_bytes() == before


async def test_cancelled_agent_partial_note_retains_provenance_without_swallowing_cancellation(auto_resource_env):
    """Cancellation propagates, but an already written image note still supports later deletion."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    written = asyncio.Event()

    class WaitingAfterWrite(FakeImageAgentWrapper):
        """Pause after the actual file job so cancellation occurs after its side effect."""

        async def reply(self, inputs, **kwargs):
            await super().reply(inputs, **kwargs)
            written.set()
            await asyncio.Event().wait()

    wrapper = WaitingAfterWrite(caption_fields("photo", "Photo", "Partial but valid caption."))
    step = env.processor(wrapper)
    task = asyncio.create_task(env.run(step, [{"change": "added", "path": str(source)}]))
    try:
        await asyncio.wait_for(written.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    note = env.workspace / "daily/2026-01-01/photo.md"
    assert frontmatter.load(note)["source_resource"] == "[[resource/2026-01-01/photo.png]]"
    assert step.context.response.metadata["modified"] is True
    deleted = await env.run(step, [{"change": "deleted", "path": str(source)}])
    assert deleted.success
    assert not note.exists()
    assert len(wrapper.calls) == 1
