"""Client projection preserves live/history semantics without exposing provider records."""

# pylint: disable=missing-function-docstring

from pathlib import Path
import runpy

import pytest

from reme.schema import SessionMessage, StreamChunk
from reme.utils.agent_transcript import TranscriptProjector, project_messages, tool_actions


def project(*chunks, completed=None):
    projector = TranscriptProjector("session", "turn")
    for chunk in chunks:
        projector.update(StreamChunk.model_validate(chunk))
    return projector.message(completed=completed)


def test_null_ends_empty_replacements_final_corrections_and_late_deltas():
    projector = TranscriptProjector("session", "turn")
    for text in ("你", "好"):
        projector.update(StreamChunk(block_id="intro", chunk=text))
    projector.update(StreamChunk(block_id="intro", chunk=None, event="end"))
    projector.update(StreamChunk(block_id="intro", chunk="stale"))
    projector.update(StreamChunk(block_id="answer", chunk="Draft", event="start"))
    projector.update(StreamChunk(block_id="answer", chunk="", event="end"))
    projector.update(StreamChunk(chunk_type="reply_end", status="completed", chunk={"result": "Corrected"}))
    message = projector.message()
    assert message.content == "你好\n\nCorrected"
    assert [item.id for item in message.items] == ["intro", "answer"]
    assert message.items[-1].phase == "final_answer"
    assert message.completed and message.outcome == "completed"


COMMAND = {
    "type": "commandExecution",
    "command": "cat note.md; ls; rg topic; npm test",
    "commandActions": [
        {"type": "read", "path": "note.md"},
        {"type": "listFiles", "path": "daily"},
        {"type": "search", "path": "daily", "query": "topic"},
        {"type": "unknown", "command": "npm test"},
    ],
}


@pytest.mark.parametrize("output", ["Complete output", None, ""])
def test_native_tool_history_equals_live_projection_and_keeps_action_targets(output):
    final = {
        "chunk_type": "tool_result",
        "tool_call_id": "call",
        "tool_call_name": "commandExecution",
        "event": "end",
        "status": "completed",
        "chunk": {**COMMAND, "aggregatedOutput": output},
    }
    history = project(final)
    live = project(
        {**final, "chunk_type": "tool_call", "event": "start", "status": "running", "chunk": COMMAND},
        {"chunk_type": "tool_result", "tool_call_id": "call", "chunk": "Partial output"},
        final,
    )
    assert history == live
    tool = history.items[0]
    assert tool.input == COMMAND["command"] and tool.output == (output or "")
    assert [action.kind for action in tool.actions] == ["read", "list", "search", "command"]
    assert tool.actions[0].target == "note.md" and tool.actions[2].query == "topic"


def test_interleaved_tools_stay_in_order_and_tool_failure_does_not_fail_the_turn():
    reply = project(
        {"chunk_type": "tool_call", "tool_call_id": "first", "chunk": {"command": "one"}},
        {"chunk_type": "content", "block_id": "text", "chunk": "Progress", "event": "end"},
        {"chunk_type": "tool_call", "tool_call_id": "second", "chunk": {"command": "two"}},
        {"chunk_type": "tool_result", "tool_call_id": "first", "chunk": {"success": False}, "event": "end"},
        {"chunk_type": "reply_end", "status": "completed", "chunk": {"result": "Done"}},
    )
    assert [item.id for item in reply.items] == ["first", "text", "second"]
    assert [item.status for item in reply.items] == ["failed", "completed", "unknown"]
    assert reply.outcome == "completed" and reply.error is None


@pytest.mark.parametrize(
    "name,payload,kind,target,query",
    [
        ("fileChange", {"type": "fileChange", "changes": [{"path": "a.md"}]}, "edit", "a.md", None),
        ("functions.view_image", {"path": "chart.png"}, "image", "chart.png", None),
        ("read_file", {"arguments": '{"path":"daily/report.md"}'}, "read", "daily/report.md", None),
        ("exec_command", {"cmd": "cat README.md"}, "command", "cat README.md", None),
        (
            "webSearch",
            {"type": "webSearch", "action": {"type": "search", "query": "Android", "queries": []}},
            "webSearch",
            None,
            "Android",
        ),
        (
            "webSearch",
            {"type": "webSearch", "action": {"type": "openPage", "url": "https://example.com"}},
            "browse",
            "https://example.com",
            None,
        ),
        (
            "webSearch",
            {"type": "webSearch", "action": {"type": "findInPage", "url": "https://example.com", "pattern": "ARM64"}},
            "search",
            "https://example.com",
            "ARM64",
        ),
    ],
)
def test_tool_semantics_are_normalized_at_the_backend(name, payload, kind, target, query):
    action = tool_actions(name, payload)[0]
    assert action["kind"] == kind and action.get("target") == target and action.get("query") == query


def test_patches_mcp_results_and_public_attachments_hide_native_diagnostics():
    reply = project(
        {
            "chunk_type": "tool_result",
            "tool_call_id": "read",
            "event": "end",
            "chunk": {"type": "mcpToolCall", "arguments": {"path": "a.md"}, "result": None, "error": None},
        },
        {
            "chunk_type": "tool_result",
            "tool_call_id": "edit",
            "event": "end",
            "chunk": {"type": "fileChange", "changes": [{"path": "a.md", "diff": "+ Updated"}]},
        },
        {
            "chunk_type": "think",
            "block_id": "private",
            "chunk": "secret",
            "metadata": {"native": {"secret": "provider"}},
        },
        {"chunk_type": "think", "block_id": "public", "event": "end", "chunk": "Plan", "metadata": {"kind": "plan"}},
        {"chunk_type": "data", "block_id": "image", "media_type": "image/png", "chunk": "aGVsbG8="},
    )
    assert reply.items[0].input == '{\n  "path": "a.md"\n}' and reply.items[0].output == ""
    assert reply.items[1].input == "a.md\n+ Updated" and reply.items[1].output == ""
    assert reply.items[2].type == "summary" and reply.items[2].kind == "plan"
    assert reply.items[3].url == "data:image/png;base64,aGVsbG8=" and reply.items[3].label == "Image"
    assert "secret" not in reply.model_dump_json() and "native" not in reply.model_dump_json()


def test_grouped_history_matches_live_identity_without_mutating_native_messages():
    native = [
        SessionMessage(
            id="model-1",
            role="assistant",
            content="Progress",
            turn_id="turn",
            chunks=[StreamChunk(block_id="intro", chunk="Progress", event="end")],
        ),
        SessionMessage(
            id="model-2",
            role="assistant",
            content="Answer",
            turn_id="turn",
            completed=False,
            chunks=[StreamChunk(block_id="answer", chunk="Answer")],
        ),
        SessionMessage(id="reminder", role="assistant", source="proactive", content="Reminder", turn_id="turn"),
    ]
    before = [item.model_dump() for item in native]
    messages = project_messages("session", native, notifications=True)
    assert [item.id for item in messages] == ["turn:assistant", "reminder"]
    assert messages[0].content == "Progress\n\nAnswer" and not messages[0].completed
    assert [item.id for item in messages[0].items] == ["intro", "answer"]
    assert messages[1].notification_eligible
    assert before == [item.model_dump() for item in native]


def test_generated_client_contract_matches_public_schemas_and_real_projection():
    root = Path(__file__).resolve().parents[2]
    generator = runpy.run_path(str(root / "reme_desktop/scripts/client-contract.py"))
    assert (root / "reme_desktop/src/shared/wire.ts").read_text() == generator["render"]()
    assert (root / "reme_desktop/tests/contract-fixture.ts").read_text() == generator["fixture"]()
