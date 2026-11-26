"""Native tool accounting includes calls omitted from typed SDK history."""

# pylint: disable=missing-function-docstring

import json
from types import SimpleNamespace

import pytest

from reme.components.agent_wrapper.codex_session_history import read_tool_calls
from reme.components.codex_app_server._session_history import item_chunks
from reme.enumeration import ChunkEnum


@pytest.mark.parametrize("kind", ["imageView", "imageGeneration", "clock", "sleep"])
def test_native_tools_have_stable_call_and_result_ids(kind):
    item = SimpleNamespace(id="call-1", type=kind, model_dump=lambda **_: {"id": "call-1", "type": kind})
    start = item_chunks(item, event="start")[0]
    end = item_chunks(item, event="end")[0]
    assert start.chunk_type == ChunkEnum.TOOL_CALL and end.chunk_type == ChunkEnum.TOOL_RESULT
    assert start.tool_call_id == end.tool_call_id == "call-1"


def test_raw_turn_calls_include_rejected_tools_and_native_search_actions(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    path = directory / "rollout-2026-11-19-session-1.jsonl"
    records = [
        {"type": "turn_context", "payload": {"turn_id": "older"}},
        {"type": "response_item", "payload": {"type": "function_call", "call_id": "old", "name": "exec_command"}},
        {"type": "turn_context", "payload": {"turn_id": "current"}},
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "search",
                "name": "exec_command",
                "arguments": '{"cmd":"rg error daily"}',
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {"id": "search", "parsed_cmd": [{"type": "search", "query": "error", "path": "daily"}]},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "rejected",
                "name": "apply_patch",
                "input": "invalid patch",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "image",
                "name": "view_image",
                "arguments": '{"path":"image.png"}',
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "web_search_call",
                "id": "web",
                "action": {"type": "search", "queries": ["one", "two"]},
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + '{"unfinished":')
    calls = read_tool_calls(path, tmp_path, "current")
    assert [call["id"] for call in calls] == ["search", "rejected", "image", "web"]
    assert calls[0]["actions"] == [{"type": "search", "query": "error", "path": "daily"}]
    assert calls[1]["arguments"] == "invalid patch"
    assert calls[-1]["name"] == "web_search"
    assert len(read_tool_calls(path, tmp_path)) == 5
    assert read_tool_calls(tmp_path / "not-created.jsonl", tmp_path) is None
    assert read_tool_calls(None, tmp_path) is None


def test_tool_statistics_preserve_history_containment(tmp_path):
    with pytest.raises(ValueError, match="outside its configured home"):
        read_tool_calls(tmp_path.parent / "escape.jsonl", tmp_path)


def test_nested_tools_are_counted_even_without_response_item_calls(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    records = [
        {"type": "turn_context", "payload": {"turn_id": "current"}},
        {
            "type": "response_item",
            "payload": {"type": "custom_tool_call", "call_id": "parent", "name": "exec", "input": "run tools"},
        },
        *[
            {"type": "event_msg", "payload": {"type": "item_completed", "item": item}}
            for item in [
                {"type": "ImageView", "id": "image", "path": "file:///screen.png"},
                {"type": "Clock", "id": "clock"},
                {
                    "type": "CommandExecution",
                    "id": "search",
                    "command": "cd daily; rg status . | head",
                    "parsed_cmd": [{"type": "unknown"}],
                },
                {"type": "Reasoning", "id": "reasoning"},
            ]
        ],
    ]
    path = directory / "rollout-2026-11-19-session-1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    calls = read_tool_calls(path, tmp_path, "current")
    assert [c["id"] for c in calls] == ["parent", "image", "clock", "search"]
    assert calls[-1]["arguments"]["command"] == "cd daily; rg status . | head"
