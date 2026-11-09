"""A quiet decision must not hide tool failures or unwanted send attempts."""

# pylint: disable=missing-function-docstring,protected-access

from benchmark.proactive.cases import MEMORY, frozen_now
from benchmark.proactive.reminder_quality import checks, scenarios


def record():
    return {
        "before": {MEMORY: "confirmed memory"},
        "after": {MEMORY: "confirmed memory"},
        "jobs": [
            {
                "name": name,
                "args": {"content": "", "kind": "normal"} if name == "send_chat_message" else {},
                "response": {"success": True},
            }
            for name in (
                "read_conversation",
                "read_proactive_inbox",
                "read_cron_results",
                "get_chat_messages",
                "send_chat_message",
            )
        ],
        "tool_events": [],
        "delivered": [],
        "completed": True,
    }


def test_empty_submission_is_successful_silence():
    verdict = checks(scenarios(frozen_now())["proactive_seen_page"], record())
    assert verdict["delivery"] and verdict["send_attempts"] and verdict["required_tools"]


def test_silence_requires_reading_conversation_in_the_merged_input_query():
    result = record()
    result["jobs"] = [job for job in result["jobs"] if job["name"] != "read_conversation"]
    assert not checks(scenarios(frozen_now())["proactive_seen_page"], result)["required_tools"]


def test_native_rejected_call_is_not_hidden_by_no_delivery():
    result = record()
    result["tool_events"] = [{"event": "start", "tool_call_name": "mcp__reme_jobs__send_chat_message", "chunk": {}}]
    assert not checks(scenarios(frozen_now())["proactive_seen_page"], result)["send_attempts"]


def test_tool_rejection_fails_protocol_and_error_checks():
    result = record()
    result["jobs"] = []
    result["tool_events"] = [{"chunk_type": "tool_result", "chunk": {"error": {"message": "requires approval"}}}]
    verdict = checks(scenarios(frozen_now())["proactive_seen_page"], result)
    assert not verdict["required_tools"] and not verdict["no_tool_errors"]


def test_awaited_result_requires_actual_delivery_and_nonempty_submission():
    verdict = checks(scenarios(frozen_now())["proactive_new"], record())
    assert not verdict["delivery"] and not verdict["send_attempts"]


def test_proactive_analysis_preserves_long_term_memory():
    result = record()
    result["after"][MEMORY] = "inferred interest"
    assert not checks(scenarios(frozen_now())["proactive_seen_reply"], result)["memory_preserved"]


def test_unrequested_new_files_and_rejected_oversize_attempts_are_observable():
    result = record()
    result["after"]["daily/unrequested.md"] = "unrequested observation"
    result["tool_events"] = [
        {
            "event": "start",
            "tool_call_name": "send_chat_message",
            "chunk": {"arguments": {"content": "x" * 4001}},
        },
    ]
    verdict = checks(scenarios(frozen_now())["proactive_seen_page"], result)
    assert not verdict["no_unexpected_files"] and not verdict["no_oversize_attempt"]
