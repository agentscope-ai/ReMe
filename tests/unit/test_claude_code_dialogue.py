"""Claude native block projection and the shared, durable Harness delivery contract."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from base64 import b64encode
import asyncio
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest

from reme.config import resolve_app_config
from reme.components.job import CronJob
from reme.schema import ResourceMetadata
from reme.steps.proactive.connector.harness import ClaudeCodeDialogueInitStep, ClaudeCodeDialogueStep
from reme.steps.proactive.connector.harness._claude_code_index import ClaudeCodeDialogueIndex
from reme.steps.proactive.connector.harness import base as harness_base
from reme.utils.files.workspace_files import WorkspaceFiles
from .proactive_helpers import running_system

NOW = datetime(2026, 11, 27, tzinfo=timezone.utc)
SESSION = "a71c17e3-b981-40a7-b30f-33d2ef709240"


def record(role="user", text="Question", identifier="user-1", *, parts=None, stop=None, block=0, **extra):
    raw = {"role": role, "content": parts if parts is not None else [{"type": "text", "text": text}]}
    if role == "assistant":
        raw.update(id=f"msg-{identifier}", stop_reason=stop)
    return {
        "type": role,
        "uuid": f"{identifier}-{block}",
        "sessionId": SESSION,
        "timestamp": (NOW + timedelta(seconds=1)).isoformat(),
        "cwd": "/projects/ReMe2",
        "apiBlockIndex": block,
        "message": raw,
        **extra,
    }


def append(path, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode())


@pytest.fixture
def source(tmp_path):
    home = tmp_path / "claude"
    path = home / "projects/project" / f"{SESSION}.jsonl"
    append(path, record(text="Old", identifier="old", timestamp=(NOW - timedelta(days=1)).isoformat()))
    return home, path


def index(tmp_path, home):
    files = WorkspaceFiles(tmp_path / "workspace")
    return ClaudeCodeDialogueIndex(files, files.path("resource/claude_code_dialogue"), home / "projects", lambda: None)


def scan(current, **kwargs):
    result = current.scan(
        current.load(NOW),
        kwargs.get("read_limit", 32 * 1024 * 1024),
        kwargs.get("output_limit", 65536),
    )
    current.commit(current.load(NOW), {"id": str(len(list(current.files.iter_jsonl(current.progress_path)))), **result})
    return result


def test_groups_api_blocks_and_keeps_real_user_turn(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(
        path,
        record(),
        {"type": "system", "subtype": "local_command", "message": "Native system text"},
        record("assistant", "Working", "progress"),
        record("assistant", identifier="progress", parts=[{"type": "thinking", "thinking": "Hidden"}], block=1),
        record(
            "assistant",
            identifier="progress",
            parts=[{"type": "tool_use", "id": "tool"}],
            stop="tool_use",
            block=2,
        ),
        record(parts=[{"type": "tool_result", "tool_use_id": "tool", "content": "Hidden"}]),
        record("assistant", "Complete", "answer", stop="end_turn"),
    )
    result = scan(current)
    assert not result["errors"]
    assert [row["content"] for row in result["messages"]] == ["Question", "Working", "Complete"]
    assert [row.get("phase") for row in result["messages"]] == [None, "commentary", "final_answer"]
    assert {row["turn_id"] for row in result["messages"]} == {"user-1-0"}
    assert [row["message_id"] for row in result["messages"]] == ["user-1-0", "msg-progress", "msg-answer"]
    assert not scan(index(tmp_path, home))["messages"]


def test_unfinished_message_survives_scan_and_restart(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(path, record(), record("assistant", "First block", "answer"))
    assert [row["role"] for row in scan(current)["messages"]] == ["user"]
    before = current.progress_path.read_bytes()
    assert not scan(index(tmp_path, home))["messages"]
    assert current.progress_path.read_bytes() == before
    append(path, record("assistant", "Second block", "answer", block=1, stop="end_turn"))
    result = scan(index(tmp_path, home))
    assert [row["content"] for row in result["messages"]] == ["First block\nSecond block"]
    assert result["messages"][0]["phase"] == "final_answer"


def test_bounded_pass_keeps_pending_blocks_and_turn_context(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(path, record(), record("assistant", "Progress", "answer"))
    rows = []
    for _ in range(8):
        result = scan(index(tmp_path, home), read_limit=1)
        rows.extend(result["messages"])
    append(path, record("assistant", "Final", "answer", block=1, stop="end_turn"))
    rows.extend(scan(index(tmp_path, home), read_limit=1)["messages"])
    assert [row["content"] for row in rows] == ["Question", "Progress\nFinal"]
    assert rows[-1]["turn_id"] == "user-1-0"


@pytest.mark.parametrize(
    "stop,phase",
    [("end_turn", "final_answer"), ("tool_use", "commentary"), ("max_tokens", "commentary")],
)
def test_only_natural_turn_completion_is_final(tmp_path, source, stop, phase):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(path, record("assistant", "Answer", "answer", stop=stop))
    message = scan(current)["messages"][0]
    assert message["phase"] == phase and message["turn_id"] == "old-0"


@pytest.mark.parametrize("flag", ["isSidechain", "isMeta", "isCompactSummary", "isSynthetic", "teamName"])
def test_filters_non_user_records(tmp_path, source, flag):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(path, record(**{flag: True}), record("assistant", stop="end_turn", **{flag: True}))
    result = scan(current)
    assert not result["messages"] and result["scanned_bytes"]


def test_filters_commands_injected_prompts_and_nested_subagents(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(path, record(text="<command-name>/help</command-name>"), record(origin={"kind": "task_notification"}))
    append(path.parent / SESSION / "subagents/agent-test.jsonl", record())
    append(path.parent / "session-index.jsonl", record())
    assert not scan(current)["messages"]


def test_images_are_embedded_bounded_and_never_read_from_arbitrary_paths(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    image = b"embedded-image"
    append(
        path,
        record(
            parts=[
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64encode(image).decode()},
                },
                {"type": "image", "source": {"type": "url", "url": "file:///etc/passwd"}},
            ],
        ),
    )
    result = scan(current)
    assert list(result["images"].values()) == [image]
    assert result["messages"][0]["image_attachments"] == 2
    assert len(result["messages"][0]["image_paths"]) == 1


def test_image_paths_are_included_in_output_byte_budget(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    state = current.load(NOW)
    append(
        path,
        record(
            parts=[
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64encode(b"image").decode()},
                },
            ],
        ),
    )
    full = current.scan(state, 65536, 65536)
    encoded_size = len((json.dumps(full["messages"][0], ensure_ascii=False) + "\n").encode())
    limit = encoded_size + sum(map(len, full["images"].values())) - 1
    bounded = current.scan(state, 65536, limit)
    assert not bounded["errors"] and not bounded["images"]
    message = bounded["messages"][0]
    assert message["image_attachments"] == 1 and message["image_paths"] == []
    assert len((json.dumps(message, ensure_ascii=False) + "\n").encode()) <= limit


def test_partial_record_and_replacement_are_retryable_without_duplicates(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    raw = json.dumps(record(text="你好"), ensure_ascii=False).encode()
    with path.open("ab") as handle:
        handle.write(raw[:-1])
    assert not scan(current)["messages"]
    with path.open("ab") as handle:
        handle.write(raw[-1:] + b"\n")
    assert len(scan(current)["messages"]) == 1
    replacement = path.with_suffix(".new")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    append(path, record(text="Next", identifier="next"))
    assert [row["content"] for row in scan(current)["messages"]] == ["Next"]


def test_output_budget_retries_final_group_at_unchanged_eof(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(path, record(), record("assistant", "Final", "answer", stop="end_turn"))
    first = scan(current, output_limit=450)
    assert [row["role"] for row in first["messages"]] == ["user"]
    second = scan(index(tmp_path, home), output_limit=450)
    assert [row["content"] for row in second["messages"]] == ["Final"]
    assert not scan(current)["messages"]


def test_malformed_source_does_not_block_other_sessions(tmp_path, source):
    home, path = source
    current = index(tmp_path, home)
    current.load(NOW)
    append(path, record(message={"content": [123]}))
    healthy = path.with_name("ed56dca5-9c60-4b44-9874-d63c3a00b516.jsonl")
    append(healthy, record(text="Healthy", identifier="healthy"))
    result = scan(current)
    assert list(result["errors"]) == [path.relative_to(home / "projects").as_posix()]
    assert [row["content"] for row in result["messages"]] == ["Healthy"]


@pytest.mark.asyncio
async def test_configured_time_initializes_before_polling_and_delivers_both_targets(tmp_path, source):
    home, path = source
    async with running_system(
        tmp_path / "workspace",
        runtimes=[
            {"agent_type": "event", "agent_name": "proactive_memory"},
            {"agent_type": "event", "agent_name": "proactive_note"},
        ],
    ) as (manager, context, _):
        jobs = resolve_app_config(config="proactive", log_config=False)["jobs"]
        config = jobs["claude_code_dialogue_cron"]
        assert config["steps"][0]["runtime_names"] == jobs["codex_dialogue_cron"]["steps"][0]["runtime_names"]
        for key in ("init_steps", "steps"):
            config[key][0]["claude_config_dir"] = str(home)
        job = CronJob(name="claude_code_dialogue_cron", app_context=context, **config)
        waits = 0

        async def wait(_delay):
            nonlocal waits
            waits += 1
            if waits == 1:
                assert (tmp_path / "workspace/resource/claude_code_dialogue/progress.jsonl").is_file()
                append(path, record(timestamp=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()))
            else:
                job._stop_event.set()

        job._next_fire_delay = lambda: 0
        job._wait_or_stop = wait
        async with job:
            await asyncio.wait_for(job._task, 2)
        for name in config["steps"][0]["runtime_names"]:
            assert (await (await manager.resolve(name)).inbox.snapshot())["pending_count"] == 1


@pytest.mark.asyncio
async def test_publishes_same_resource_contract_and_delivers_both_tasks(tmp_path, source, monkeypatch):
    home, path = source
    clock = Mock(wraps=datetime)
    clock.now.return_value = NOW
    monkeypatch.setattr(harness_base, "datetime", clock)
    async with running_system(
        tmp_path / "workspace",
        language="zh",
        runtimes=[
            {"agent_type": "event", "agent_name": "proactive_memory"},
            {"agent_type": "event", "agent_name": "proactive_note"},
        ],
    ) as (manager, context, _):
        config = resolve_app_config(config="proactive", log_config=False)["jobs"]["claude_code_dialogue_cron"]
        options = {**config["steps"][0], "app_context": context, "claude_config_dir": home}
        await ClaudeCodeDialogueInitStep(**options)()
        append(path, record(), record("assistant", "完成", "answer", stop="end_turn"))
        response = await ClaudeCodeDialogueStep(**options)()
        assert response.success and set(response.answer) == {
            "event_ids",
            "dialogue_messages",
            "scanned_bytes",
            "errors",
        }
        assert response.answer["dialogue_messages"] == 2
        for name in ("proactive_memory", "proactive_note"):
            events = (await (await manager.resolve(name)).inbox.claim())["items"]
            assert [event["id"] for event in events] == response.answer["event_ids"]
            assert events[0]["source"] == "claude_code_dialogue"
            paths = [tmp_path / "workspace" / value for value in events[0]["paths"]]
            metadata = ResourceMetadata.model_validate_json(paths[0].read_text())
            assert {entry.path for entry in metadata.files} == {"dialogue.md", "dialogue.jsonl"}
            assert "Claude Code 最终回答" in paths[1].read_text()
        assert not (await ClaudeCodeDialogueStep(**options)()).answer["event_ids"]
