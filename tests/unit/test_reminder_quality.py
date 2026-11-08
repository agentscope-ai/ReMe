"""A quiet decision must not hide tool failures or unwanted send attempts."""

# pylint: disable=missing-function-docstring,protected-access

from benchmark.proactive.cases import MEMORY
from benchmark.proactive.reminder_quality import checks, scenarios


def record():
    return {
        "before": {MEMORY: "confirmed memory"},
        "after": {MEMORY: "confirmed memory"},
        "jobs": [
            {
                "name": name,
                "args": {"content": ""} if name == "send_reminder" else {},
                "response": {"success": True},
            }
            for name in (
                "get_proactive_messages",
                "read_conversation",
                "read_task_results",
                "get_reminder",
                "send_reminder",
            )
        ],
        "tool_events": [],
        "delivered": [],
        "completed": True,
    }


def test_empty_submission_is_successful_silence():
    verdict = checks(scenarios()["proactive_seen_page"], record(), False)
    assert verdict["delivery"] and verdict["send_attempts"] and verdict["six_query_protocol"]


def test_native_rejected_call_is_not_hidden_by_no_delivery():
    result = record()
    result["tool_events"] = [{"event": "start", "tool_call_name": "mcp__reme_jobs__send_reminder", "chunk": {}}]
    assert not checks(scenarios()["proactive_seen_page"], result, False)["send_attempts"]


def test_tool_rejection_fails_protocol_and_error_checks():
    result = record()
    result["jobs"] = []
    result["tool_events"] = [{"chunk_type": "tool_result", "chunk": {"error": {"message": "requires approval"}}}]
    verdict = checks(scenarios()["proactive_seen_page"], result, False)
    assert not verdict["six_query_protocol"] and not verdict["no_tool_errors"]


def test_awaited_result_requires_actual_delivery_and_nonempty_submission():
    verdict = checks(scenarios()["proactive_awaited_result"], record(), True)
    assert not verdict["delivery"] and not verdict["send_attempts"]


def test_proactive_analysis_preserves_long_term_memory():
    result = record()
    result["after"][MEMORY] = "inferred interest"
    assert not checks(scenarios()["proactive_seen_reply"], result, False)["memory_preserved"]


def test_unrequested_new_files_and_rejected_oversize_attempts_are_observable():
    result = record()
    result["after"]["daily/unrequested.md"] = "unrequested observation"
    result["tool_events"] = [
        {
            "event": "start",
            "tool_call_name": "send_reminder",
            "chunk": {"arguments": {"content": "x" * 201}},
        },
    ]
    verdict = checks(scenarios()["proactive_seen_page"], result, False)
    assert not verdict["no_unexpected_files"] and not verdict["no_oversize_attempt"]
