"""Tests for answer_reach_limit in AsAgentWrapper and CcAgentWrapper.

Covers all listed cases:

AS (AsAgentWrapper):
  A1  – normal path (no exceed), answer_reach_limit=True
  A2  – exceed + forced answer success
  A3  – answer_reach_limit=False + exceed
  A4  – _call_model raises → graceful fallback
  A5  – _call_model returns None
  A6  – _call_model returns empty text blocks
  A7  – _call_model returns async generator
  A8  – output_schema sees forced answer in context

CC (CcAgentWrapper):
  C1  – normal path (no exceed), answer_reach_limit=True
  C2  – exceed + forced answer success
  C3  – answer_reach_limit=False + exceed
  C4  – _force_answer raises → graceful fallback
  C5  – _force_answer returns None
  C6  – _force_answer returns empty string
  C7  – no session_id on last_msg
  C8  – trailing error + answer_reach_limit=True → swallow
  C9  – trailing error + answer_reach_limit=False → raise
  C10 – no messages received → ValueError

Helpers:
  H1  – _is_max_turns_result
  H2  – _deny_all_tools hook structure
"""

# pylint: disable=protected-access,missing-function-docstring,missing-class-docstring

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _make_text_block(text: str):
    """Return a minimal TextBlock-like object."""
    from agentscope.message import TextBlock

    return TextBlock(type="text", text=text)


def _make_tool_block():
    """Return a non-text block (simulates a tool_use block)."""
    # A plain object with type != "text" so getattr(b, "type", None) != "text"
    m = MagicMock()
    m.type = "tool_use"
    m.text = "should-be-ignored"
    return m


# ===========================================================================
# AS (AsAgentWrapper) tests
# ===========================================================================


class _FakeReactConfig:
    def __init__(self, max_iters: int = 3):
        self.max_iters = max_iters


class _FakeState:
    def __init__(self, cur_iter: int = 0, session_id: str = "sess-001"):
        self.cur_iter = cur_iter
        self.session_id = session_id
        self.context: list = []


class _FakeFinalMsg:
    """Mimics AgentScope final message returned by agent.reply()."""

    def __init__(self, text: str = "placeholder text"):
        self._text = text

    def get_text_content(self) -> str:
        return self._text

    def model_dump(self) -> dict:
        return {"role": "assistant", "content": [{"type": "text", "text": self._text}]}


class _FakeModelResponse:
    """Mimics a ChatResponse from AgentScope model."""

    def __init__(self, blocks: list | None = None):
        self.content = blocks if blocks is not None else [_make_text_block("forced answer")]


def _build_as_wrapper():
    """Return an AsAgentWrapper with heavy internals stubbed out."""
    from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper

    wrapper = AsAgentWrapper.__new__(AsAgentWrapper)
    # Stub logger
    wrapper.logger = MagicMock()
    # Stub _dump_state so it doesn't touch disk
    wrapper._dump_state = AsyncMock()
    # Stub as_llm (not needed for answer_reach_limit tests unless output_schema)
    wrapper.as_llm = None
    # kwargs (for _merged_kwargs)
    wrapper.kwargs = {}
    return wrapper


def _setup_agent_mock(
    wrapper,
    cur_iter: int = 0,
    max_iters: int = 3,
    final_text: str = "placeholder text",
):
    """Create a mock agent and patch _build_agent to return it.

    Returns (agent_mock, state, patcher) — caller must start/stop patcher.
    """
    state = _FakeState(cur_iter=cur_iter)
    react_config = _FakeReactConfig(max_iters=max_iters)
    final_msg = _FakeFinalMsg(text=final_text)

    agent = MagicMock()
    agent.state = state
    agent.react_config = react_config
    agent.name = "test_agent"
    agent.observe = AsyncMock()
    agent.reply = AsyncMock(return_value=final_msg)

    # _prepare_model_input
    agent._prepare_model_input = AsyncMock(
        return_value={
            "messages": [MagicMock()],  # some existing message
            "tools": [MagicMock()],
        },
    )

    # _call_model: default to a valid response
    agent._call_model = AsyncMock(
        return_value=_FakeModelResponse(),
    )

    patcher = patch.object(
        type(wrapper),
        "_build_agent",
        new_callable=AsyncMock,
        return_value=(agent, "processed_inputs"),
    )
    return agent, state, patcher


# ---- A1: normal path (no exceed) ----


@pytest.mark.asyncio
async def test_as_a1_no_exceed():
    """answer_reach_limit=True but cur_iter < max_iters → normal result."""
    wrapper = _build_as_wrapper()
    agent, state, patcher = _setup_agent_mock(wrapper, cur_iter=1, max_iters=3)

    with patcher:
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "placeholder text"
    assert result["session_id"] == "sess-001"
    agent._call_model.assert_not_called()
    wrapper._dump_state.assert_awaited_once_with(state)


# ---- A2: exceed + forced answer success ----


@pytest.mark.asyncio
async def test_as_a2_exceed_forced_success():
    """cur_iter == max_iters + forced answer → result is forced text."""
    wrapper = _build_as_wrapper()
    _agent, state, patcher = _setup_agent_mock(
        wrapper,
        cur_iter=3,
        max_iters=3,
        final_text="Executed maximum iterations...",
    )

    with patcher:
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "forced answer"
    # last_message content should be updated to forced text
    assert result["last_message"]["content"][0]["text"] == "forced answer"
    # forced answer should be persisted into context
    assert len(state.context) == 1
    wrapper._dump_state.assert_awaited_once_with(state)


# ---- A3: answer_reach_limit=False + exceed ----


@pytest.mark.asyncio
async def test_as_a3_no_flag_exceed():
    """answer_reach_limit=False + exceed → no forced call, returns placeholder."""
    wrapper = _build_as_wrapper()
    agent, _state, patcher = _setup_agent_mock(
        wrapper,
        cur_iter=3,
        max_iters=3,
        final_text="Executed maximum iterations...",
    )

    with patcher:
        result = await wrapper.reply("hello", answer_reach_limit=False)

    assert result["result"] == "Executed maximum iterations..."
    agent._call_model.assert_not_called()


# ---- A4: _call_model raises → graceful fallback ----


@pytest.mark.asyncio
async def test_as_a4_call_model_raises():
    """_call_model raises → falls back to original result."""
    wrapper = _build_as_wrapper()
    agent, _state, patcher = _setup_agent_mock(
        wrapper,
        cur_iter=3,
        max_iters=3,
        final_text="original placeholder",
    )
    agent._call_model = AsyncMock(side_effect=RuntimeError("model boom"))

    with patcher:
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "original placeholder"
    wrapper.logger.warning.assert_called_once()


# ---- A5: _call_model returns None ----


@pytest.mark.asyncio
async def test_as_a5_call_model_returns_none():
    """_call_model returns None → no forced answer."""
    wrapper = _build_as_wrapper()
    agent, state, patcher = _setup_agent_mock(
        wrapper,
        cur_iter=3,
        max_iters=3,
        final_text="original",
    )
    agent._call_model = AsyncMock(return_value=None)

    with patcher:
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "original"
    assert len(state.context) == 0


# ---- A6: _call_model returns empty text blocks ----


@pytest.mark.asyncio
async def test_as_a6_empty_text_blocks():
    """_call_model returns only non-text blocks → no forced answer."""
    wrapper = _build_as_wrapper()
    agent, state, patcher = _setup_agent_mock(
        wrapper,
        cur_iter=3,
        max_iters=3,
        final_text="original",
    )
    agent._call_model = AsyncMock(
        return_value=_FakeModelResponse(blocks=[_make_tool_block()]),
    )

    with patcher:
        result = await wrapper.reply("hello", answer_reach_limit=True)

    # forced_text is "" (falsy) → forced_answer stays False
    assert result["result"] == "original"
    assert len(state.context) == 0


# ---- A7: _call_model returns async generator ----


@pytest.mark.asyncio
async def test_as_a7_async_generator_response():
    """_call_model returns an async generator → collects last chunk."""
    wrapper = _build_as_wrapper()
    agent, state, patcher = _setup_agent_mock(
        wrapper,
        cur_iter=3,
        max_iters=3,
        final_text="original",
    )

    async def _fake_stream():
        yield _FakeModelResponse(blocks=[_make_text_block("chunk1")])
        yield _FakeModelResponse(blocks=[_make_text_block("chunk2-final")])

    agent._call_model = AsyncMock(return_value=_fake_stream())

    with patcher:
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "chunk2-final"
    assert len(state.context) == 1


# ---- A8: output_schema sees forced answer in context ----


@pytest.mark.asyncio
async def test_as_a8_output_schema_with_forced():
    """Forced answer is appended to context before output_schema call."""
    wrapper = _build_as_wrapper()
    _agent, state, patcher = _setup_agent_mock(
        wrapper,
        cur_iter=3,
        max_iters=3,
        final_text="placeholder",
    )

    # Fake as_llm with a model that has generate_structured_output
    fake_model = MagicMock()
    fake_model.generate_structured_output = AsyncMock(
        return_value=MagicMock(content={"key": "value"}),
    )
    wrapper.as_llm = MagicMock()
    wrapper.as_llm.model = fake_model

    with patcher:
        result = await wrapper.reply(
            "hello",
            answer_reach_limit=True,
            output_schema={"type": "object"},
        )

    # Forced answer should be in context
    assert len(state.context) == 1

    # generate_structured_output should be called with context that includes forced answer
    fake_model.generate_structured_output.assert_awaited_once()
    call_kwargs = fake_model.generate_structured_output.call_args
    assert call_kwargs.kwargs["messages"] is state.context

    assert result["structured_output"] == {"key": "value"}


# ===========================================================================
# CC (CcAgentWrapper) tests
# ===========================================================================


@dataclass
class _FakeResultMessage:
    """Mimics claude_agent_sdk.ResultMessage (dataclass)."""

    result: str = "normal answer"
    session_id: str | None = "sess-cc-001"
    subtype: str | None = "success"
    is_error: bool = False
    structured_output: Any = None


def _build_cc_wrapper():
    """Return a CcAgentWrapper with heavy internals stubbed out."""
    from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

    wrapper = CcAgentWrapper.__new__(CcAgentWrapper)
    wrapper.logger = MagicMock()
    wrapper.kwargs = {}
    # _build_options returns a mock opts
    wrapper._build_options = MagicMock(return_value=MagicMock())
    return wrapper


def _patch_cc_query(messages: list, *, error_after: bool = False):
    """Patch both ``claude_agent_sdk.query`` and ``claude_agent_sdk.ResultMessage``.

    ``ResultMessage`` is patched with ``_FakeResultMessage`` so that the
    ``isinstance(msg, ResultMessage)`` check inside ``reply()`` succeeds.

    When *error_after* is True, the query generator raises RuntimeError
    after yielding all messages (simulates CLI trailing exit error).
    """

    async def _gen(*_args, **_kwargs):
        for msg in messages:
            yield msg
        if error_after:
            raise RuntimeError("CLI exit error")

    return patch.multiple(
        "claude_agent_sdk",
        query=MagicMock(side_effect=_gen),
        ResultMessage=_FakeResultMessage,
    )


# ---- C1: normal path (no exceed) ----


@pytest.mark.asyncio
async def test_cc_c1_no_exceed():
    """answer_reach_limit=True but subtype=success → normal result."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(result="normal answer", subtype="success")

    with _patch_cc_query([msg]):
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "normal answer"
    assert result["session_id"] == "sess-cc-001"


# ---- C2: exceed + forced answer success ----


@pytest.mark.asyncio
async def test_cc_c2_exceed_forced_success():
    """error_max_turns + forced success → result is forced text."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(
        result="max turns reached",
        subtype="error_max_turns",
        is_error=True,
    )

    with (
        _patch_cc_query([msg]),
        patch.object(
            type(wrapper),
            "_force_answer",
            new_callable=AsyncMock,
            return_value="forced cc answer",
        ),
    ):
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "forced cc answer"
    assert result["last_message"]["result"] == "forced cc answer"


# ---- C3: answer_reach_limit=False + exceed ----


@pytest.mark.asyncio
async def test_cc_c3_no_flag_exceed():
    """answer_reach_limit=False + exceed → returns original result."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(
        result="max turns reached",
        subtype="error_max_turns",
        is_error=True,
    )

    with _patch_cc_query([msg]):
        result = await wrapper.reply("hello", answer_reach_limit=False)

    assert result["result"] == "max turns reached"


# ---- C4: _force_answer raises → graceful fallback ----


@pytest.mark.asyncio
async def test_cc_c4_force_answer_raises():
    """_force_answer raises → falls back to original result."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(
        result="original cc result",
        subtype="error_max_turns",
        is_error=True,
    )

    with (
        _patch_cc_query([msg]),
        patch.object(
            type(wrapper),
            "_force_answer",
            new_callable=AsyncMock,
            side_effect=RuntimeError("force boom"),
        ),
    ):
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "original cc result"
    wrapper.logger.warning.assert_called_once()


# ---- C5: _force_answer returns None ----


@pytest.mark.asyncio
async def test_cc_c5_force_answer_returns_none():
    """_force_answer returns None → no forced answer."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(
        result="original",
        subtype="error_max_turns",
        is_error=True,
    )

    with (
        _patch_cc_query([msg]),
        patch.object(
            type(wrapper),
            "_force_answer",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "original"


# ---- C6: _force_answer returns empty string ----


@pytest.mark.asyncio
async def test_cc_c6_force_answer_returns_empty():
    """_force_answer returns '' → falsy, no forced answer."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(
        result="original",
        subtype="error_max_turns",
        is_error=True,
    )

    with (
        _patch_cc_query([msg]),
        patch.object(
            type(wrapper),
            "_force_answer",
            new_callable=AsyncMock,
            return_value="",
        ),
    ):
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "original"


# ---- C7: no session_id on last_msg ----


@pytest.mark.asyncio
async def test_cc_c7_no_session_id():
    """error_max_turns but session_id is None → skip forced answer."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(
        result="no session",
        subtype="error_max_turns",
        is_error=True,
        session_id=None,
    )

    with _patch_cc_query([msg]):
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "no session"
    assert result["session_id"] == ""


# ---- C8: trailing error + answer_reach_limit=True → swallow ----


@pytest.mark.asyncio
async def test_cc_c8_trailing_error_swallowed():
    """query() raises after ResultMessage + answer_reach_limit=True → swallowed."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(result="ok result", subtype="success")

    with _patch_cc_query([msg], error_after=True):
        result = await wrapper.reply("hello", answer_reach_limit=True)

    assert result["result"] == "ok result"


# ---- C9: trailing error + answer_reach_limit=False → raise ----


@pytest.mark.asyncio
async def test_cc_c9_trailing_error_raised():
    """query() raises after ResultMessage + answer_reach_limit=False → re-raises."""
    wrapper = _build_cc_wrapper()
    msg = _FakeResultMessage(result="ok result", subtype="success")

    with _patch_cc_query([msg], error_after=True):
        with pytest.raises(RuntimeError, match="CLI exit error"):
            await wrapper.reply("hello", answer_reach_limit=False)


# ---- C10: no messages received → ValueError ----


@pytest.mark.asyncio
async def test_cc_c10_no_messages():
    """query() yields nothing → ValueError."""
    wrapper = _build_cc_wrapper()

    with _patch_cc_query([]):
        with pytest.raises(ValueError, match="No message received"):
            await wrapper.reply("hello")


# ===========================================================================
# CC helper method tests
# ===========================================================================


# ---- H1: _is_max_turns_result ----


class TestIsMaxTurnsResult:
    def test_error_max_turns(self):
        from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

        msg = _FakeResultMessage(subtype="error_max_turns")
        assert CcAgentWrapper._is_max_turns_result(msg) is True

    def test_success_subtype(self):
        from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

        msg = _FakeResultMessage(subtype="success")
        assert CcAgentWrapper._is_max_turns_result(msg) is False

    def test_none_subtype(self):
        from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

        msg = _FakeResultMessage(subtype=None)
        assert CcAgentWrapper._is_max_turns_result(msg) is False

    def test_no_subtype_attr(self):
        from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

        # Object without subtype attribute
        obj = MagicMock(spec=[])  # empty spec → no attributes
        assert CcAgentWrapper._is_max_turns_result(obj) is False

    def test_non_string_subtype(self):
        from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

        msg = MagicMock()
        msg.subtype = 42
        assert CcAgentWrapper._is_max_turns_result(msg) is False


# ---- H2: _deny_all_tools hook ----


@pytest.mark.asyncio
async def test_deny_all_tools_structure():
    """_deny_all_tools returns correct hook output dict."""
    from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

    result = await CcAgentWrapper._deny_all_tools(
        _input={"tool_name": "Bash"},
        _tool_use_id="tool-123",
        _context=None,
    )

    hook_output = result["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "PreToolUse"
    assert hook_output["permissionDecision"] == "deny"
    assert "permissionDecisionReason" in hook_output
    assert isinstance(hook_output["permissionDecisionReason"], str)
