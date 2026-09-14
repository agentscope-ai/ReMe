"""Direct image inputs preserve native model binding and session contracts."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
import base64
import copy
import io
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from agentscope.agent import Agent, ContextConfig
from agentscope.message import Base64Source, Msg, TextBlock, URLSource
from PIL import Image
import pytest
import yaml

from reme.components.as_llm import BaseAsLLM
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
from reme.steps.evolve import _auto_memory_image, auto_image_resource

from .test_auto_memory_image_modes import (
    _DAY,
    _SESSION,
    _DirectModel,
    _Harness,
    _MemoryWrapper,
    _main_saved_line,
    _message,
    _png,
)


@pytest.fixture(name="harness")
def direct_harness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return _Harness(tmp_path, monkeypatch)


async def _run(harness, messages, **kwargs):
    step = harness.step()
    await step(session_id=_SESSION, date=_DAY, messages=messages, include_images=True, **kwargs)
    return step.context.response


def test_default_config_keeps_images_disabled_and_selects_direct_mode():
    path = Path(__file__).resolve().parents[2] / "reme" / "config" / "default.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = config["jobs"]["auto_memory"]
    properties = job["parameters"]["properties"]
    assert properties["include_images"]["default"] is False
    assert properties["supports_vision"]["default"] is False
    assert properties["image_mode"]["default"] == "direct"
    assert properties["image_mode"]["enum"] == ["direct"]
    assert job["steps"][0]["include_images"] is False
    assert job["steps"][0]["supports_vision"] is False
    assert job["steps"][0]["image_mode"] == "direct"
    assert [step["backend"] for step in job["steps"]] == ["auto_memory_step", "auto_tag_step"]
    assert "enable_tags" not in job["steps"][0]
    assert job["steps"][1] == {"backend": "auto_tag_step"}


@pytest.mark.asyncio
async def test_enabled_default_is_direct_with_a_single_memory_reply(harness, monkeypatch):
    caption = AsyncMock(side_effect=AssertionError("Direct mode must not caption images"))
    monkeypatch.setattr(auto_image_resource.AutoImageResourceStep, "_caption_with_retry", caption)
    message = _message()
    message.content[0].text += " Keep literal [Image 1]; image ID: gallery-before."
    message.content.insert(2, TextBlock(text="Between observations; image ID: gallery-after."))
    message.content.append(TextBlock(text="After both observations."))
    before = copy.deepcopy(message.model_dump())

    response = await _run(harness, [message])

    assert response.success is True
    assert response.metadata["auto_memory_images"] == {
        "mode": "direct",
        "status": "completed",
        "image_count": 2,
        "captioned_images": 0,
    }
    assert len(harness.wrapper.calls) == 1
    inputs, kwargs = harness.wrapper.calls[0]
    assert isinstance(inputs, Msg)
    assert inputs.role == "user"
    assert [block.type for block in inputs.content] == ["text", "data", "text", "data", "text"]
    assert "Remember this observation." in inputs.content[0].text
    assert "Keep literal [Image 1]; image ID: gallery-before." in inputs.content[0].text
    assert inputs.content[2].text.strip() == "Between observations; image ID: gallery-after."
    assert "After both observations." in inputs.content[4].text
    assert "# Your Task" in inputs.content[4].text
    assert inputs.content[4].text.index("After both observations.") < inputs.content[4].text.index("# Your Task")
    assert inputs.get_text_content().count("[Image 1]") == 1
    assert "[Image 2]" not in inputs.get_text_content()
    assert "__reme_image_" not in inputs.get_text_content()
    assert "TOOL_RESULT_MUST_NOT_BE_SAVED" not in inputs.get_text_content()
    assert "Caption (model-generated)" not in inputs.get_text_content()
    assert "image captions" not in kwargs["system_prompt"].lower()
    assert inputs.content[1].model_dump() == message.content[1].model_dump()
    assert inputs.content[3].model_dump() == message.content[3].model_dump()
    assert message.model_dump() == before
    assert harness.session_path.read_bytes() == _main_saved_line(message)
    assert not (harness.workspace / "resource").exists()
    caption.assert_not_called()
    assert "_model" not in kwargs
    assert "supports_vision" not in kwargs

    formatted = await harness.wrapper.as_llm.model.formatter.format([inputs])
    assert len(formatted) == 1
    assert [part["type"] for part in formatted[0]["content"]] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert [part["text"] for part in formatted[0]["content"] if part["type"] == "text"] == [
        block.text for block in inputs.content if block.type == "text"
    ]
    image_parts = [part for part in formatted[0]["content"] if part["type"] == "image_url"]
    assert [part["image_url"]["url"] for part in image_parts] == [
        f"data:image/png;base64,{message.content[index].source.data}" for index in (1, 3)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_images", [None, False])
async def test_disabled_keeps_original_string_and_does_not_prepare(harness, monkeypatch, include_images):
    prepare = AsyncMock(side_effect=AssertionError("Disabled images must not be inspected"))
    monkeypatch.setattr("reme.steps.evolve.auto_memory.prepare_direct_message", prepare)
    step = harness.step()
    kwargs = {} if include_images is None else {"include_images": include_images}
    message = _message()
    message.content[1].source.data = "invalid-base64"

    await step(session_id=_SESSION, date=_DAY, messages=[message], image_mode="future-mode", **kwargs)

    assert step.context.response.success is True
    assert isinstance(harness.wrapper.calls[0][0], str)
    assert "[Image" not in harness.wrapper.calls[0][0]
    assert "auto_memory_images" not in step.context.response.metadata
    assert "_model" not in harness.wrapper.calls[0][1]
    assert harness.session_path.read_bytes() == _main_saved_line(message)
    prepare.assert_not_called()


@pytest.mark.asyncio
async def test_no_images_skips_source_checks(harness, monkeypatch):
    loader = AsyncMock(side_effect=AssertionError("No image should be loaded"))
    monkeypatch.setattr(_auto_memory_image, "_image_bytes", loader)

    response = await _run(harness, [_message(images=False)])

    assert response.success is True
    assert isinstance(harness.wrapper.calls[0][0], str)
    assert response.metadata["auto_memory_images"]["status"] == "skipped"
    assert response.metadata["auto_memory_images"]["image_count"] == 0
    assert "_model" not in harness.wrapper.calls[0][1]
    harness.logger.warning.assert_not_called()
    loader.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("images,supported", [(True, True), (True, False), (False, False)])
async def test_non_agentscope_wrapper_is_rejected_even_without_images_or_support(
    harness,
    monkeypatch,
    images,
    supported,
):
    harness.wrapper = _MemoryWrapper()
    harness.wrapper.backend = "agentscope"
    loader = AsyncMock(side_effect=AssertionError("Backend rejection must precede image IO"))
    monkeypatch.setattr(_auto_memory_image, "_image_bytes", loader)

    with pytest.raises(NotImplementedError, match="AgentScope"):
        await _run(harness, [_message(images=images)], supports_vision=supported)

    assert not harness.wrapper.calls
    harness.logger.warning.assert_not_called()
    loader.assert_not_called()


@pytest.mark.asyncio
async def test_non_agentscope_wrapper_still_accepts_original_text_only_calls(harness):
    harness.wrapper = _MemoryWrapper()
    step = harness.step()

    await step(session_id=_SESSION, date=_DAY, messages=[_message()], include_images=False)

    assert step.context.response.success is True
    assert isinstance(harness.wrapper.calls[0][0], str)
    assert "auto_memory_images" not in step.context.response.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("declared,images", [(None, True), (False, True), (None, False)])
async def test_missing_vision_declaration_does_not_read_images(
    harness,
    monkeypatch,
    declared,
    images,
):
    loader = AsyncMock(side_effect=AssertionError("Undeclared support must not load images"))
    monkeypatch.setattr(_auto_memory_image, "_image_bytes", loader)
    step = harness.step()
    step.kwargs.pop("supports_vision")
    message = _message(images=images)
    if images:
        message.content[1].source.data = "invalid-base64"
    before = message.model_dump()
    options = {} if declared is None else {"supports_vision": declared}

    await step(session_id=_SESSION, date=_DAY, messages=[message], include_images=True, **options)

    response = step.context.response
    assert response.success is True
    assert response.metadata["auto_memory_images"] == {
        "mode": "direct",
        "status": "fallback" if images else "skipped",
        "image_count": 2 if images else 0,
        "captioned_images": 0,
        **({"reason": "supports_vision=false"} if images else {}),
    }
    inputs, kwargs = harness.wrapper.calls[0]
    assert isinstance(inputs, str)
    assert "Remember this observation." in inputs
    assert "_model" not in kwargs
    assert "context_config" not in kwargs
    assert "supports_vision" not in kwargs
    assert harness.logger.warning.call_count == int(images)
    assert message.model_dump() == before
    assert harness.session_path.read_bytes() == _main_saved_line(message)
    loader.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_images", [False, True])
async def test_memory_uses_bound_as_llm_even_when_another_vision_component_exists(harness, include_images):
    models = harness.app_context.components[ComponentEnum.AS_LLM]
    bound = BaseAsLLM()
    bound.model = _DirectModel("bound-memory")
    vision = BaseAsLLM()
    vision.model = _DirectModel("unused-resource-vision")
    models.update(custom=bound, vision=vision)
    harness.wrapper.as_llm = bound
    step = harness.step()

    await step(session_id=_SESSION, date=_DAY, messages=[_message()], include_images=include_images)

    inputs, kwargs = harness.wrapper.calls[0]
    assert isinstance(inputs, Msg if include_images else str)
    assert "_model" not in kwargs and "supports_vision" not in kwargs
    # Exercise native Agent construction without resolving the test's file tools.
    agent, _ = await harness.wrapper._build_agent(inputs, **{**kwargs, "job_tools": []})
    assert agent.model is bound.model
    assert harness.wrapper.as_llm is bound
    assert models["default"] is not bound
    assert models["vision"] is vision


def _six_image_message():
    message = _message()
    message.content = [message.content[0], *[message.content[1].model_copy(deep=True) for _ in range(6)]]
    return message


@pytest.mark.asyncio
async def test_default_direct_preserves_six_images_through_real_sdk_context_limiting(harness):
    original_context = {"trigger_ratio": 0.7, "reserve_ratio": 0.2}
    harness.wrapper.kwargs["context_config"] = original_context

    response = await _run(harness, [_six_image_message()])

    assert response.metadata["auto_memory_images"]["status"] == "completed"
    assert response.metadata["auto_memory_images"]["image_count"] == 6
    inputs, options = harness.wrapper.calls[0]
    config = options["context_config"]
    assert config == {"trigger_ratio": 0.7, "reserve_ratio": 0.2, "max_image_num": 6}
    assert config is not original_context
    assert original_context == {"trigger_ratio": 0.7, "reserve_ratio": 0.2}
    assert harness.wrapper.kwargs["context_config"] is original_context
    agent = Agent(
        name="memory-context-check",
        system_prompt="Record useful visual facts.",
        model=harness.wrapper.as_llm.model,
        context_config=ContextConfig(**config),
    )
    await agent.observe(inputs)
    await agent._limit_context_images(agent.context_config)
    assert sum(block.type == "data" for message in agent.state.context for block in message.content) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 5, 6, 10])
async def test_explicit_image_limits_are_respected_without_partial_image_inputs(harness, limit):
    context_config = {"max_image_num": limit, "trigger_ratio": 0.7}
    harness.wrapper.kwargs["context_config"] = context_config

    response = await _run(harness, [_six_image_message()])

    expected = "fallback" if limit < 6 else "completed"
    assert response.metadata["auto_memory_images"]["status"] == expected
    assert isinstance(harness.wrapper.calls[0][0], str if limit < 6 else Msg)
    assert context_config == {"max_image_num": limit, "trigger_ratio": 0.7}
    if limit < 6:
        assert response.metadata["auto_memory_images"]["reason"] == "context-image-limit: ValueError"
        assert "context_config" not in harness.wrapper.calls[0][1]
        assert "_model" not in harness.wrapper.calls[0][1]
        harness.logger.warning.assert_called_once()
    else:
        assert harness.wrapper.calls[0][1]["context_config"] == context_config


@pytest.mark.asyncio
@pytest.mark.parametrize("override_limit", [None, 0, 10])
async def test_reply_context_override_has_shallow_precedence_without_mutating_source(
    harness,
    monkeypatch,
    override_limit,
):
    component_config = {"max_image_num": 5, "trigger_ratio": 0.7}
    harness.wrapper.kwargs["context_config"] = component_config
    call_config = {"reserve_ratio": 0.2}
    if override_limit is not None:
        call_config["max_image_num"] = override_limit
    original = dict(call_config)
    step = harness.step()
    hook = Mock(return_value={"context_config": call_config})
    monkeypatch.setattr(step, "_reply_extra_kwargs", hook)

    await step(session_id=_SESSION, date=_DAY, messages=[_six_image_message()], include_images=True)

    expected = "fallback" if override_limit == 0 else "completed"
    assert step.context.response.metadata["auto_memory_images"]["status"] == expected
    assert component_config == {"max_image_num": 5, "trigger_ratio": 0.7}
    assert call_config == original
    effective = harness.wrapper.calls[0][1]["context_config"]
    assert "trigger_ratio" not in effective
    assert effective["max_image_num"] == (6 if override_limit is None else override_limit)
    hook.assert_called_once_with(_DAY)


@pytest.mark.asyncio
async def test_interleaving_preserves_whole_history_hook_and_image_only_turn_with_duplicate_ids(harness, monkeypatch):
    first = _message()
    first.content.insert(2, TextBlock(text="This separates the two observations."))
    second = _message("second", f"{_DAY}T11:00:00", images=False)
    second.name = "user"
    second.role = "user"
    second.content = [first.content[1].model_copy(deep=True)]
    messages = [first, second]
    originals = [message.model_dump() for message in messages]
    step = harness.step()

    def format_whole_history(history):
        # Like LME/BEAM, the hook depends on the full slice, not one Msg per call.
        return f"Source excerpt: L41-L{40 + len(history)}\n" + "\n\n".join(
            f"[L{line} | {message.name} @ {message.created_at}]\n{message.get_text_content()}"
            for line, message in enumerate(history, start=41)
        )

    hook = Mock(side_effect=format_whole_history)
    monkeypatch.setattr(step, "_format_history", hook)

    await step(session_id=_SESSION, date=_DAY, messages=messages, include_images=True)

    inputs = harness.wrapper.calls[0][0]
    assert step.context.response.metadata["auto_memory_images"]["status"] == "completed"
    assert hook.call_count == 2  # Original fallback prompt, then the temporary multimodal history.
    assert all(len(call.args[0]) == 2 for call in hook.call_args_list)
    assert inputs.get_text_content().count("Source excerpt: L41-L42") == 1
    assert "[L41 | assistant @" in inputs.content[0].text
    assert inputs.content[2].text.strip() == "This separates the two observations."
    assert f"[L42 | user @ {second.created_at}]" in inputs.content[4].text
    assert "# Your Task" in inputs.content[-1].text
    assert "__reme_image_" not in inputs.get_text_content()
    assert "[Image" not in inputs.get_text_content()
    assert [block.type for block in inputs.content] == ["text", "data", "text", "data", "text", "data", "text"]
    data_blocks = [block for block in inputs.content if block.type == "data"]
    assert len(data_blocks) == 3
    assert len({block.id for block in data_blocks}) == 1
    assert data_blocks[0].source.data == data_blocks[2].source.data
    assert data_blocks[0].source.data != data_blocks[1].source.data
    assert [message.model_dump() for message in messages] == originals
    assert harness.session_path.read_bytes() == b"".join(_main_saved_line(message) for message in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("language,existing", [("en", False), ("zh", False), ("en", True), ("zh", True)])
async def test_interleaving_keeps_localized_create_update_prompt_boundaries(harness, monkeypatch, language, existing):
    step = harness.step(language=language)
    note_path = f"daily/{_DAY}/existing.md"
    if existing:
        monkeypatch.setattr(step, "_list_session_note", AsyncMock(return_value={"path": note_path}))
        monkeypatch.setattr(step, "_ensure_session_frontmatter", AsyncMock())
        monkeypatch.setattr(step, "_rename_from_frontmatter_name", AsyncMock(return_value=note_path))
        monkeypatch.setattr("reme.steps.evolve.auto_memory.refresh_day_index", AsyncMock(return_value={}))
    message = _message()

    await step(session_id=_SESSION, date=_DAY, messages=[message], include_images=True)

    inputs, options = harness.wrapper.calls[0]
    assert step.context.response.success is True
    assert step.context.response.metadata["auto_memory_images"]["status"] == "completed"
    assert inputs.content[0].text.startswith("Today:" if language == "en" else "今天：")
    assert "Remember this observation." in inputs.content[0].text
    assert ("# Your Task" if language == "en" else "# 你的任务") in inputs.content[-1].text
    assert ("frontmatter_update" if existing else "daily_write") in inputs.content[-1].text
    assert [block.id for block in inputs.content if block.type == "data"] == [message.content[1].id] * 2
    assert "__reme_image_" not in inputs.get_text_content()
    assert "[Image" not in inputs.get_text_content()
    if existing:
        assert note_path in inputs.content[0].text
        assert options["injected_job_kwargs"] == {"_allowed_paths": [note_path]}


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["No history requested.", "{history}\n{history}"])
async def test_missing_or_repeated_history_falls_back_without_direct_overrides(harness, monkeypatch, template):
    step = harness.step(prompt_dict={"user_message_create": template})
    original_options = {"context_config": {"trigger_ratio": 0.7}, "model_config": {"max_retries": 2}}
    monkeypatch.setattr(step, "_reply_extra_kwargs", Mock(return_value=original_options))
    message = _message()

    await step(session_id=_SESSION, date=_DAY, messages=[message], include_images=True)

    response = step.context.response
    inputs, options = harness.wrapper.calls[0]
    assert response.success is True
    assert response.metadata["auto_memory_images"]["status"] == "fallback"
    assert response.metadata["auto_memory_images"]["reason"] == "prompt: ValueError"
    assert inputs == template.format(history=step._format_history([message]))
    assert "__reme_image_" not in inputs
    assert "_model" not in options
    assert options["context_config"] is original_options["context_config"]
    assert original_options == {"context_config": {"trigger_ratio": 0.7}, "model_config": {"max_retries": 2}}
    harness.logger.warning.assert_called_once()
    assert harness.session_path.read_bytes() == _main_saved_line(message)


@pytest.mark.asyncio
async def test_invalid_original_prompt_remains_an_error_before_image_preparation(harness, monkeypatch):
    prepare = AsyncMock(side_effect=AssertionError("Invalid templates must fail before image processing"))
    monkeypatch.setattr("reme.steps.evolve.auto_memory.prepare_direct_message", prepare)
    step = harness.step(prompt_dict={"user_message_create": "{missing_variable}"})

    with pytest.raises(KeyError, match="missing_variable"):
        await step(session_id=_SESSION, date=_DAY, messages=[_message()], include_images=True)

    prepare.assert_not_called()
    harness.logger.warning.assert_not_called()
    assert not harness.wrapper.calls


@pytest.mark.asyncio
async def test_direct_uses_shared_normalized_provider_bytes(harness):
    with Image.new("RGB", (2300, 8), (20, 40, 220)) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="BMP")
    original_data = buffer.getvalue()
    message = _message()
    message.content = [message.content[0], message.content[1]]
    message.content[1].source = Base64Source(
        media_type="image/bmp",
        data=base64.b64encode(original_data).decode("ascii"),
    )
    before = message.model_dump()
    step = harness.step()
    step.context = RuntimeContext()
    expected = auto_image_resource._build_image_request_payload(original_data, "")

    direct_message = await _auto_memory_image.prepare_direct_message(step, [message], step._format_history)

    image_input = next(block.source for block in direct_message.content if block.type == "data")
    assert image_input.media_type == expected["mime"] == "image/jpeg"
    assert image_input.data == expected["data_b64"]
    with Image.open(io.BytesIO(base64.b64decode(image_input.data))) as image:
        assert max(image.size) == auto_image_resource.MAX_IMAGE_REQUEST_DIMENSION
    assert message.model_dump() == before


@pytest.mark.asyncio
async def test_workspace_file_is_bounded_and_materialized_before_sdk_input(harness):
    source = harness.workspace / "original image.png"
    source.write_bytes(_png())
    message = _message()
    message.content = [message.content[0], message.content[1]]
    message.content[1].source = URLSource(media_type="image/png", url=source.as_uri())
    before = message.model_dump()

    response = await _run(harness, [message])

    assert response.metadata["auto_memory_images"]["status"] == "completed"
    source_input = next(block.source for block in harness.wrapper.calls[0][0].content if block.type == "data")
    assert source_input.type == "base64"
    assert base64.b64decode(source_input.data) == source.read_bytes() == _png()
    assert message.model_dump() == before
    assert harness.session_path.read_bytes() == _main_saved_line(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid-base64", "invalid-image", "outside-workspace", "permission"])
async def test_second_image_failure_discards_all_prepared_inputs(harness, failure):
    message = _six_image_message()
    source_config = {"trigger_ratio": 0.7}
    harness.wrapper.kwargs["context_config"] = source_config
    kwargs = {}
    if failure == "invalid-base64":
        message.content[2].source.data = "INVALID_BASE64!!!"
    elif failure == "invalid-image":
        message.content[2].source.data = base64.b64encode(b"not an image").decode("ascii")
    elif failure == "outside-workspace":
        source = harness.workspace.parent / "out-of-workspace-image.png"
        source.write_bytes(_png())
        message.content[2].source = URLSource(media_type="image/png", url=source.as_uri())
    else:
        source = harness.workspace / "restricted-image.png"
        source.write_bytes(_png())
        message.content[2].source = URLSource(media_type="image/png", url=source.as_uri())
        kwargs["_allowed_paths"] = ["allowed-image.png"]
    before = message.model_dump()

    response = await _run(harness, [message], **kwargs)

    assert response.success is True
    assert response.metadata["auto_memory_images"]["status"] == "fallback"
    assert isinstance(harness.wrapper.calls[0][0], str)
    assert "_model" not in harness.wrapper.calls[0][1]
    assert "[Image" not in harness.wrapper.calls[0][0]
    assert "Remember this observation." in harness.wrapper.calls[0][0]
    assert response.metadata["auto_memory_images"]["captioned_images"] == 0
    assert "context_config" not in harness.wrapper.calls[0][1]
    assert source_config == {"trigger_ratio": 0.7}
    harness.logger.warning.assert_called_once()
    assert message.model_dump() == before
    assert harness.session_path.read_bytes() == _main_saved_line(message)


@pytest.mark.asyncio
async def test_preparation_cancellation_propagates_without_memory_reply(harness, monkeypatch):
    loader = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(_auto_memory_image, "_image_bytes", loader)

    with pytest.raises(asyncio.CancelledError):
        await _run(harness, [_message()])

    assert not harness.wrapper.calls
    harness.logger.warning.assert_not_called()
    assert harness.session_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_fails", [False, True])
async def test_memory_reply_error_is_never_retried_as_text(harness, image_fails):
    harness.wrapper.error = RuntimeError("Failure after a possible memory-tool write")
    message = _message()
    if image_fails:
        message.content[1].source.data = "invalid-base64"

    with pytest.raises(RuntimeError, match="possible memory-tool write"):
        await _run(harness, [message])

    assert len(harness.wrapper.calls) == 1
    assert isinstance(harness.wrapper.calls[0][0], str if image_fails else Msg)
    assert harness.logger.warning.call_count == int(image_fails)
    assert harness.session_path.exists()
