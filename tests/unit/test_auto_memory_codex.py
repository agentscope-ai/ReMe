"""Unit tests for the auto_memory_codex step."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json
from pathlib import Path

import pytest

from reme.steps.evolve.auto_memory_codex import AutoMemoryCodexStep

# ---- helpers ---------------------------------------------------------------


def _msg(role: str, text: str | list | None = None, *, pid: str = "m1") -> dict:
    """Codex response_item with a ``message`` payload."""
    if text is None:
        text = "hello"
    content = text if isinstance(text, list) else [{"type": "input_text", "text": text}]
    return {
        "type": "response_item",
        "payload": {"type": "message", "id": pid, "role": role, "content": content},
    }


def _fc(name: str, arguments: str = "{}", *, cid: str = "c1") -> dict:
    """Codex response_item with a ``function_call`` payload."""
    return {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "id": cid,
            "name": name,
            "arguments": arguments,
            "call_id": cid,
        },
    }


def _fco(cid: str, output: str) -> dict:
    """Codex response_item with a ``function_call_output`` payload."""
    return {
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": cid, "output": output},
    }


# ---- _codex_entries_to_messages --------------------------------------------


class TestEntriesToMessages:
    def test_user_and_assistant(self):
        msgs = AutoMemoryCodexStep._codex_entries_to_messages(
            [_msg("user", "hi", pid="1"), _msg("assistant", "hey", pid="2")],
        )
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "hi"

    def test_filters_system_and_developer_roles(self):
        msgs = AutoMemoryCodexStep._codex_entries_to_messages(
            [_msg("system", pid="1"), _msg("developer", pid="2"), _msg("user", "real", pid="3")],
        )
        assert len(msgs) == 1
        assert msgs[0]["content"] == "real"

    def test_skips_non_response_item_rows(self):
        msgs = AutoMemoryCodexStep._codex_entries_to_messages(
            [{"type": "session_meta", "payload": {"id": "s1"}}, _msg("user", "real", pid="1")],
        )
        assert len(msgs) == 1

    def test_function_call_and_output(self):
        msgs = AutoMemoryCodexStep._codex_entries_to_messages(
            [_fc("shell", json.dumps({"cmd": "ls"}), cid="c1"), _fco("c1", "file1\nfile2")],
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert "[tool shell" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "[tool_result" in msgs[1]["content"]
        assert "file1" in msgs[1]["content"]

    def test_custom_tool_call(self):
        entry = {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "id": "c1",
                "name": "mcp_search",
                "input": '{"q":"x"}',
                "call_id": "c1",
            },
        }
        msgs = AutoMemoryCodexStep._codex_entries_to_messages([entry])
        assert len(msgs) == 1
        assert "mcp_search" in msgs[0]["content"]

    def test_empty_output_skipped(self):
        assert not AutoMemoryCodexStep._codex_entries_to_messages([_fco("c1", "")])

    def test_long_arguments_truncated(self):
        entry = _fc("t", json.dumps({"d": "x" * 500}), cid="c1")
        msgs = AutoMemoryCodexStep._codex_entries_to_messages([entry])
        assert msgs[0]["content"].endswith("...)]")

    def test_reasoning_skipped(self):
        msgs = AutoMemoryCodexStep._codex_entries_to_messages(
            [{"type": "response_item", "payload": {"type": "reasoning", "id": "r1"}}, _msg("user", "real", pid="1")],
        )
        assert len(msgs) == 1
        assert msgs[0]["content"] == "real"

    def test_realistic_multi_turn_session(self):
        """Parse a realistic Codex transcript with mixed entry types."""
        transcript = [
            # session_meta — skipped
            {
                "timestamp": "2026-07-20T10:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "sess-abc",
                    "cwd": "/project",
                    "originator": "codex_exec",
                    "cli_version": "0.144.0",
                    "source": "exec",
                    "timestamp": "2026-07-20T10:00:00.000Z",
                },
            },
            # turn_context — skipped
            {
                "timestamp": "2026-07-20T10:00:01.000Z",
                "type": "turn_context",
                "payload": {
                    "turn_id": "turn-1",
                    "cwd": "/project",
                    "model": "gpt-5.1",
                },
            },
            # user message
            {
                "timestamp": "2026-07-20T10:00:02.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "帮我看看入口文件是什么"}],
                },
            },
            # assistant text
            {
                "timestamp": "2026-07-20T10:00:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-2",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "让我查一下。"}],
                },
            },
            # shell function_call
            {
                "timestamp": "2026-07-20T10:00:04.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "call-1",
                    "name": "shell",
                    "arguments": '{"command": "ls *.py"}',
                    "call_id": "call-1",
                },
            },
            # shell output
            {
                "timestamp": "2026-07-20T10:00:05.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "main.py\nutils.py",
                },
            },
            # assistant explains result
            {
                "timestamp": "2026-07-20T10:00:06.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-3",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "入口是 main.py。"}],
                },
            },
            # reasoning — skipped
            {
                "timestamp": "2026-07-20T10:00:07.000Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "id": "reason-1",
                    "summary": [{"type": "summary_text", "text": "用户想知道入口文件"}],
                },
            },
            # mcpToolCall as function_call
            {
                "timestamp": "2026-07-20T10:00:08.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "id": "call-2",
                    "name": "read",
                    "arguments": '{"path": "daily/2026-07-20.md"}',
                    "call_id": "call-2",
                },
            },
            # MCP tool output
            {
                "timestamp": "2026-07-20T10:00:09.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-2",
                    "output": "# 2026-07-20\n- 讨论了入口文件\n- 上次提到用 Flask",
                },
            },
            # final assistant
            {
                "timestamp": "2026-07-20T10:00:10.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-4",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "根据记忆，上次也讨论过入口文件。"}],
                },
            },
        ]

        msgs = AutoMemoryCodexStep._codex_entries_to_messages(transcript)

        # session_meta, turn_context, reasoning are skipped
        # 4 messages + 2 function_calls + 2 function_call_outputs = 8
        assert len(msgs) == 8

        # Verify ordering and types
        assert msgs[0] == {"role": "user", "name": "user", "content": "帮我看看入口文件是什么"}
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "让我查一下。"
        assert "[tool shell" in msgs[2]["content"] and "ls *.py" in msgs[2]["content"]
        assert msgs[3]["role"] == "user" and "main.py" in msgs[3]["content"]
        assert msgs[4]["content"] == "入口是 main.py。"
        assert "[tool read" in msgs[5]["content"]
        assert "2026-07-20" in msgs[6]["content"]
        assert msgs[7]["content"] == "根据记忆，上次也讨论过入口文件。"


# ---- _render_codex_content -------------------------------------------------


class TestRenderContent:
    def test_input_and_output_text(self):
        content = [{"type": "input_text", "text": "hi"}, {"type": "output_text", "text": "there"}]
        assert AutoMemoryCodexStep._render_codex_content(content) == "hi\nthere"

    def test_strips_empty_and_skips_non_dict(self):
        content = [{"type": "input_text", "text": "   "}, "bare", {"type": "input_text", "text": "real"}]
        assert AutoMemoryCodexStep._render_codex_content(content) == "real"

    def test_skips_unknown_block_types(self):
        content = [{"type": "image", "data": "x"}, {"type": "output_text", "text": "ok"}]
        assert AutoMemoryCodexStep._render_codex_content(content) == "ok"


# ---- _resolve_transcript_path ----------------------------------------------


class TestResolvePath:
    @pytest.mark.asyncio
    async def test_valid_path(self, tmp_path, monkeypatch):
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        t = sessions / "rollout-20260720-abc.jsonl"
        t.write_text(json.dumps(_msg("user", pid="r1")) + "\n")
        monkeypatch.setattr(AutoMemoryCodexStep, "_codex_sessions_dir", staticmethod(lambda: sessions))
        assert await AutoMemoryCodexStep()._resolve_transcript_path(str(t), "abc") == str(t)

    @pytest.mark.asyncio
    async def test_outside_sessions_rejected(self, tmp_path, monkeypatch):
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        outside = tmp_path / "outside.jsonl"
        outside.write_text("{}")
        monkeypatch.setattr(AutoMemoryCodexStep, "_codex_sessions_dir", staticmethod(lambda: sessions))
        assert await AutoMemoryCodexStep()._resolve_transcript_path(str(outside), "abc") is None

    @pytest.mark.asyncio
    async def test_fallback_by_session_id(self, tmp_path, monkeypatch):
        sessions = tmp_path / "sessions" / "2026" / "07" / "20"
        sessions.mkdir(parents=True)
        real = sessions / "rollout-1721400000-sess-abc.jsonl"
        real.write_text(json.dumps(_msg("user", pid="r1")) + "\n")
        monkeypatch.setattr(AutoMemoryCodexStep, "_codex_sessions_dir", staticmethod(lambda: tmp_path / "sessions"))
        result = await AutoMemoryCodexStep()._resolve_transcript_path("/gone/stale.jsonl", "sess-abc")
        assert result == str(real)


# ---- _load_codex_transcript ------------------------------------------------


@pytest.mark.asyncio
async def test_load_jsonl(tmp_path: Path):
    t = tmp_path / "r.jsonl"
    t.write_text("\n".join(json.dumps(_msg("user", pid=str(i))) for i in range(2)))
    result = await AutoMemoryCodexStep()._load_codex_transcript(str(t))
    assert len(result) == 2


# ---- _save_codex_session ---------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_by_payload_id(tmp_path, monkeypatch):
    step = AutoMemoryCodexStep()
    monkeypatch.setattr(step, "_codex_store", lambda: _store(tmp_path / "codex"))
    await step._save_codex_session("s1", [_msg("user", pid="1"), _msg("user", pid="2")])
    inc = await step._save_codex_session("s1", [_msg("user", pid="2"), _msg("user", pid="3")])
    assert len(inc) == 1
    assert inc[0]["payload"]["id"] == "3"


@pytest.mark.asyncio
async def test_dedup_by_call_id(tmp_path, monkeypatch):
    step = AutoMemoryCodexStep()
    monkeypatch.setattr(step, "_codex_store", lambda: _store(tmp_path / "codex"))
    await step._save_codex_session("s1", [_fco("c1", "o1"), _fco("c2", "o2")])
    inc = await step._save_codex_session("s1", [_fco("c1", "o1-again"), _fco("c3", "o3")])
    assert len(inc) == 1
    assert inc[0]["payload"]["call_id"] == "c3"


@pytest.mark.asyncio
async def test_filters_rows_without_id_or_call_id(tmp_path, monkeypatch):
    step = AutoMemoryCodexStep()
    monkeypatch.setattr(step, "_codex_store", lambda: _store(tmp_path / "codex"))
    inc = await step._save_codex_session(
        "s1",
        [
            _msg("user", pid="1"),
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": "no id"}},
        ],
    )
    assert len(inc) == 1


def _store(root: Path):
    from reme.components.agent_wrapper import CcFileSessionStore

    return CcFileSessionStore(root)
