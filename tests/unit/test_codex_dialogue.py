"""Independent Codex and screen collectors deliver to one work Agent using isolated sources."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from base64 import b64decode
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from reme.core.context import ApplicationContext, RuntimeContext
from reme.components import LocalProactiveManager
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.agent_inbox import AgentEventInbox
from reme.utils.files.workspace_files import WorkspaceFiles
from reme.components.job import BaseJob, CronJob
from reme.config import resolve_app_config
from reme.schema import ResourceMetadata
from reme.steps.proactive.data_connectors import (
    ScreenCaptureStep,
    CodexDialogueInitStep,
    CodexDialogueStep,
)
from reme.steps.proactive.data_connectors._codex_dialogue import CodexDialogueIndex
from reme.steps.proactive.data_connectors import codex_dialogue

_NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def record(kind, text="hello", identifier="user-1", *, phase=None, when=None, thread="thread-1"):
    when = when or _NOW + timedelta(seconds=1)
    if kind == "meta":
        return {
            "type": "session_meta",
            "payload": {
                "id": thread,
                "session_id": thread,
                "source": "vscode",
                "thread_source": "user",
                "originator": "codex-tui",
                "cwd": "/projects/ReMe2",
            },
        }
    return {
        "type": "event_msg",
        "timestamp": when.isoformat(),
        "payload": {
            "type": "item_completed",
            "turn_id": "turn-1",
            "item": {
                "type": kind,
                "id": identifier,
                "content": [{"type": "text" if kind == "UserMessage" else "Text", "text": text}],
                "phase": phase,
            },
        },
    }


def append(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for value in records:
            handle.write((json.dumps(value, ensure_ascii=False) + "\n").encode())


@pytest.fixture
def source(tmp_path):
    home = tmp_path / "codex"
    path = home / "sessions/2026/01/01/rollout-first.jsonl"
    append(path, record("meta"), record("UserMessage", "old history", when=_NOW - timedelta(days=1)))
    return home, path


@pytest.fixture(autouse=True)
def devices(monkeypatch):
    clock = Mock(wraps=datetime)
    clock.now.return_value = _NOW
    monkeypatch.setattr(codex_dialogue, "datetime", clock)
    monkeypatch.setattr("reme.steps.proactive.data_connectors.base_connector.datetime", clock)

    def capture(_step, directory):
        assert directory.is_relative_to(_step.workspace_path / "resource/screenshot")
        assert not (directory / "metadata.json").exists()
        (directory / "1.png").write_bytes(b"fake-screen-1")
        (directory / "2.png").write_bytes(b"fake-screen-2")
        return [{"path": "1.png"}, {"path": "2.png"}]

    collector = Mock(side_effect=capture)

    def collect(step, directory):
        return collector(step, directory)

    monkeypatch.setattr(ScreenCaptureStep, "collect", collect)
    return collector


@asynccontextmanager
async def runtime(tmp_path, *, work_cron=None):
    context = ApplicationContext(workspace_dir=str(tmp_path / "workspace"), language="zh")
    Path(context.app_config.workspace_dir).mkdir(exist_ok=True)
    inbox = AgentEventInbox(app_context=context)
    agents = LocalProactiveManager(
        app_context=context,
        runtimes=[
            {
                "agent_name": "work",
                "task_description": "跟踪项目进展和难点",
                "cron": work_cron,
            },
            {"agent_name": "proactive_notebook", "task_description": "跟进随记"},
        ],
    )
    context.components = {
        "agent_inbox": {"default": inbox},
        "agent_wrapper": {"default": Mock(spec=BaseAgentWrapper)},
        "proactive_manager": {"default": agents},
    }
    deliver = agents.deliver_events
    delivery = AsyncMock(side_effect=deliver)

    async def capture_delivery(task_names, events):
        if events[0].source == "codex_dialogue":
            return await delivery(task_names=task_names, events=events)
        return await deliver(task_names, events)

    agents.deliver_events = capture_delivery
    async with inbox, agents:
        agents.watcher.cancel()
        await asyncio.gather(agents.watcher, return_exceptions=True)
        agents.watcher = None
        yield context, inbox, delivery


async def capture_screen(context, **kwargs):
    job = BaseJob(
        app_context=context,
        steps=[{"backend": "screen_capture_step", "task_names": ["work"], **kwargs}],
    )
    async with job:
        return await job()


def options(context, home, **kwargs):
    return {"app_context": context, "codex_home": home, **kwargs}


def workspace(context):
    return Path(context.app_config.workspace_dir)


def progress(context):
    return workspace(context) / "resource/codex_dialogue/progress.jsonl"


def dialogue_index(context, home):
    return CodexDialogueIndex(WorkspaceFiles(workspace(context)), progress(context).parent, home, lambda: None)


def messages(context):
    return [
        json.loads(line)
        for path in sorted(workspace(context).glob("resource/codex_dialogue/*/*/dialogue.jsonl"))
        for line in path.read_text().splitlines()
    ]


@pytest.mark.asyncio
async def test_independent_collectors_deliver_dialogue_and_all_screens_to_work(tmp_path, source, monkeypatch):
    monkeypatch.setattr(
        "reme.steps.proactive.data_connectors.base_connector.uuid4",
        Mock(return_value=Mock(hex="a8f3c2" + "0" * 26)),
    )
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        original_size = path.stat().st_size
        raw_user = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "AGENTS.md"}],
            },
        }
        append(
            path,
            raw_user,
            record("UserMessage", "做增量感知"),
            record("AgentMessage", "progress", "progress", phase="commentary"),
            record("AgentMessage", "partial", "partial", phase="partial_answer"),
            record("Reasoning", "hidden reasoning"),
            record("CommandExecution", "secret output"),
            record("AgentMessage", "开发完成", "final-1", phase="final_answer"),
        )
        screen = await capture_screen(context)
        assert screen.success
        response = await CodexDialogueStep(**options(context, home))()
        assert response.success and response.answer["dialogue_messages"] == 3
        assert response.answer["scanned_bytes"] == path.stat().st_size - original_size
        assert [message["content"] for message in messages(context)] == ["做增量感知", "progress", "开发完成"]
        assert [message["phase"] for message in messages(context) if message["role"] == "assistant"] == [
            "commentary",
            "final_answer",
        ]
        assert {message["cwd"] for message in messages(context)} == {"/projects/ReMe2"}
        events = (await inbox.claim("work"))["items"]
        assert [event["source"] for event in events] == ["screenshot", "codex_dialogue"]
        assert len(events[0]["paths"]) == 2
        assert delivery.await_count == 1 and len(delivery.await_args.kwargs["events"]) == 1
        for event in events:
            first_path = workspace(context) / event["paths"][0]
            manifest = first_path.with_name("metadata.json")
            assert manifest.parent == workspace(context) / "resource" / event["source"] / "2026-11-15/120000_a8f3c2"
            metadata = ResourceMetadata.model_validate_json(manifest.read_text())
            assert metadata.source == event["source"]
            if event["source"] == "screenshot":
                expected_files = {"1.png", "2.png"}
                assert {Path(path).name for path in event["paths"]} == expected_files
            else:
                assert event["paths"] == [first_path.relative_to(workspace(context)).as_posix()]
                assert first_path.name == "dialogue.md"
                markdown = first_path.read_text()
                assert markdown.index("做增量感知") < markdown.index("Codex 进度") < markdown.index("开发完成")
                expected_files = {"dialogue.md", "dialogue.jsonl"}
            assert {entry.path for entry in metadata.files} == expected_files
            assert not {Path(value).name for value in event["paths"]} & {"progress.jsonl", "pending.json"}
            assert all(not Path(value).is_absolute() and "rollout" not in value for value in event["paths"])
        assert not (workspace(context) / "resource/screenshot/pending.json").exists()


@pytest.mark.asyncio
async def test_mid_turn_inputs_keep_order_and_deliver_image_files(tmp_path, source):
    home, path = source
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/X8AAAAASUVORK5CYII="
    url = f"data:image/png;base64,{png}"
    image_input = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": url}],
        },
    }
    steer = record("UserMessage", "follow-up", "steer")
    steer["payload"]["item"]["content"] += [{"type": "local_image", "path": "/tmp/gone.png"}]
    last = record("UserMessage", "", "image-only")
    last["payload"]["item"]["content"] = [{"type": "image", "image_url": url}]
    async with runtime(tmp_path) as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(
            path,
            record("UserMessage", "start", "first"),
            record("AgentMessage", "checking", "step", phase="commentary"),
            image_input,
        )
        # Codex can flush the model-facing input before it completes the user item.
        first = await CodexDialogueStep(**options(context, home))()
        assert first.success and first.answer["dialogue_messages"] == 2
        assert [row["message_id"] for row in messages(context)] == ["first", "step"]
        append(path, steer, last, record("AgentMessage", "done", "final", phase="final_answer"))
        response = await CodexDialogueStep(**options(context, home, task_names=["work", "proactive_notebook"]))()
        assert response.success and response.answer["dialogue_messages"] == 3
        events = (await inbox.claim("proactive_notebook"))["items"]
        assert len(events) == 1
        transcript, *images = [workspace(context) / name for name in events[0]["paths"]]
        assert transcript.name == "dialogue.md"
        rows = [json.loads(line) for line in transcript.with_suffix(".jsonl").read_text().splitlines()]
        assert [row["message_id"] for row in rows] == ["steer", "image-only", "final"]
        assert {row["turn_id"] for row in messages(context)} == {"turn-1"}
        assert [row["image_paths"] for row in rows[:2]] == [[image.name] for image in images]
        assert len(images) == 2 and all(image.read_bytes() == b64decode(png) for image in images)
        markdown = transcript.read_text()
        assert markdown.count("### 轮次 `turn-1`") == 1
        assert markdown.index("`steer`") < markdown.index("follow-up") < markdown.index(f"({images[0].name})")
        assert markdown.index("`image-only`") < markdown.index(f"({images[1].name})") < markdown.index("`final`")
        manifest = transcript.with_name("metadata.json")
        assert [entry.path for entry in ResourceMetadata.model_validate_json(manifest.read_text()).files] == [
            transcript.name,
            "dialogue.jsonl",
            *(image.name for image in images),
        ]


@pytest.mark.asyncio
async def test_restart_rebuilds_cursors_from_progress_without_replaying_dialogue(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "first"))
        await CodexDialogueStep(**options(context, home))()
    append(path, record("AgentMessage", "after restart", "final", phase="final_answer"))
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        await CodexDialogueStep(**options(context, home))()
        assert sorted(value["content"] for value in messages(context)) == ["after restart", "first"]
        index = dialogue_index(context, home)
        state = index.load(_NOW)
        assert dialogue_index(context, home).load(_NOW) == state
        assert {path.name for path in progress(context).parent.iterdir() if path.is_file()} == {"progress.jsonl"}
        assert not (workspace(context) / "resource/screenshot/pending.json").exists()
        records = list(index.files.iter_jsonl(index.progress_path))
        assert [record["type"] for record in records] == ["baseline", "commit", "commit"]
        assert set(records[1]) == {"type", "id", "updates", "message_ids", "next_path"}
        await CodexDialogueInitStep(**options(context, home))()
        response = await CodexDialogueStep(**options(context, home))()
        assert response.answer["scanned_bytes"] == response.answer["dialogue_messages"] == 0


@pytest.mark.asyncio
async def test_repeated_message_ids_are_skipped_across_appends_and_restarts(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "same text", "same-id"))
        await CodexDialogueStep(**options(context, home))()
        append(path, record("UserMessage", "same text", "same-id"))
        result = await CodexDialogueStep(**options(context, home))()
        assert result.answer["dialogue_messages"] == 0
    append(path, record("UserMessage", "same text", "same-id"), record("UserMessage", "same text", "new-id"))
    async with runtime(tmp_path) as (context, _, _):
        result = await CodexDialogueStep(**options(context, home))()
        assert result.answer["dialogue_messages"] == 1
        # Package names contain random suffixes, so directory order is not message order.
        assert sorted(item["message_id"] for item in messages(context)) == ["new-id", "same-id"]


@pytest.mark.asyncio
async def test_idle_dialogue_produces_no_events_or_progress_writes(
    tmp_path,
    source,
    monkeypatch,
):
    home, path = source
    monkeypatch.setattr(ScreenCaptureStep, "collect", Mock(return_value=[]))
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        original = progress(context).read_bytes()
        result = await CodexDialogueStep(**options(context, home))()
        assert result.success and result.answer["event_ids"] == []
        assert progress(context).read_bytes() == original
        delivery.assert_not_awaited()
        assert not list(workspace(context).glob("resource/screenshot/*/*"))
        append(path, record("CommandExecution", "tool output"))
        await CodexDialogueStep(**options(context, home))()
        advanced = progress(context).read_bytes()
        assert advanced != original
        await CodexDialogueStep(**options(context, home))()
        assert progress(context).read_bytes() == advanced
        assert (await inbox.snapshot("work"))["pending_count"] == 0


@pytest.mark.asyncio
async def test_filtered_screenshot_keeps_new_dialogue_and_uses_the_configured_threshold(tmp_path, source, monkeypatch):
    home, path = source
    thresholds = []

    def filtered_capture(step, _directory):
        thresholds.append(step.similarity_threshold)
        return []

    monkeypatch.setattr(ScreenCaptureStep, "collect", filtered_capture)
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "new project progress"))
        screen = await capture_screen(context, similarity_threshold=0.98)
        assert screen.success and screen.answer == {"skipped": "no_new_data"}
        result = await CodexDialogueStep(**options(context, home))()
        assert result.success and result.answer["dialogue_messages"] == 1
        assert thresholds == [0.98]
        assert [event.source for event in delivery.await_args.kwargs["events"]] == ["codex_dialogue"]
        assert (await inbox.snapshot("work"))["pending_count"] == 1
        assert not list(workspace(context).glob("resource/screenshot/*/*"))


@pytest.mark.asyncio
async def test_progress_replays_a_journal_larger_than_the_record_limit(tmp_path, source, monkeypatch):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        index = dialogue_index(context, home)
        state = index.load(_NOW)
        monkeypatch.setattr("reme.utils.files.workspace_files.DEFAULT_MAX_FILE_BYTES", 2048)
        for number in range(30):
            index.commit(
                state,
                {"id": f"batch-{number}", "updates": {}, "message_ids": [f"thread:{number}"], "next_path": ""},
            )
        assert index.progress_path.stat().st_size > 2048
        assert index.load(_NOW) == state
        assert set(index.load(_NOW)["known_ids"]) == {f"thread:{number}" for number in range(30)}


@pytest.mark.asyncio
async def test_incomplete_progress_tail_is_ignored_and_replaced_on_commit(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "first", "first"))
        await CodexDialogueStep(**options(context, home))()
        with progress(context).open("ab") as handle:
            handle.write(b'{"type":"commit","id":"interrupted"')
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "second", "second"))
        await CodexDialogueStep(**options(context, home))()
        records = [json.loads(line) for line in progress(context).read_text().splitlines()]
        assert [record["type"] for record in records] == ["baseline", "commit", "commit"]
        assert sorted(message["content"] for message in messages(context)) == ["first", "second"]


@pytest.mark.parametrize("content", [b"", b'{"type":"baseline"', b'{"type":"commit"}\n', b"{invalid}\n"])
@pytest.mark.asyncio
async def test_invalid_progress_baseline_fails_without_resetting(tmp_path, source, devices, content):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        progress(context).write_bytes(content)
        with pytest.raises(ValueError):
            await CodexDialogueStep(**options(context, home))()
        assert progress(context).read_bytes() == content
        devices.assert_not_called()


@pytest.mark.asyncio
async def test_failed_progress_append_does_not_advance_in_memory_cursors(tmp_path, source, monkeypatch):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        index = dialogue_index(context, home)
        state = index.load(_NOW)
        before = json.loads(json.dumps(state))
        transaction = {"id": "failed", "updates": {}, "message_ids": [], "next_path": "next"}
        with monkeypatch.context() as patch:
            patch.setattr(index.files, "append_jsonl", Mock(side_effect=OSError("append failed")))
            with pytest.raises(OSError, match="append failed"):
                index.commit(state, transaction)
        assert state == before and index.load(_NOW) == before
        index.commit(state, transaction)
        assert index.load(_NOW) == state and state["last_batch"] == "failed"


@pytest.mark.asyncio
async def test_tool_only_changes_advance_the_cursor(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("CommandExecution", "large output" * 100), record("Reasoning", "private"))
        first = await CodexDialogueStep(**options(context, home))()
        second = await CodexDialogueStep(**options(context, home))()
        assert first.answer["scanned_bytes"] > 0 and first.answer["dialogue_messages"] == 0
        assert second.answer["scanned_bytes"] == 0 and messages(context) == []


@pytest.mark.asyncio
async def test_incomplete_utf8_tail_is_retried_after_the_newline(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        line = (json.dumps(record("UserMessage", "中文对话"), ensure_ascii=False) + "\n").encode()
        split = line.index("中文".encode()) + 1
        with path.open("ab") as handle:
            handle.write(line[:split])
        first = await CodexDialogueStep(**options(context, home))()
        assert first.success and first.answer["dialogue_messages"] == 0
        with path.open("ab") as handle:
            handle.write(line[split:])
        second = await CodexDialogueStep(**options(context, home))()
        assert second.success and [value["content"] for value in messages(context)] == ["中文对话"]


@pytest.mark.asyncio
async def test_new_files_in_old_directories_and_replaced_files_are_deduplicated(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "retained"))
        await CodexDialogueStep(**options(context, home))()
        path.unlink()
        append(
            path,
            record("meta"),
            record("UserMessage", "retained"),
            record("UserMessage", "new after replacement", "user-2"),
        )
        other = path.with_name("rollout-other.jsonl")
        append(other, record("meta", thread="thread-2"), record("UserMessage", "other project", "user-3"))
        response = await CodexDialogueStep(**options(context, home))()
        assert response.answer["dialogue_messages"] == 2
        assert sorted(value["content"] for value in messages(context)) == [
            "new after replacement",
            "other project",
            "retained",
        ]


@pytest.mark.asyncio
async def test_subagents_and_reme_sessions_are_excluded(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        for name, fields in (
            ("subagent", {"source": {"subagent": {"thread_spawn": {}}}}),
            ("reme", {"originator": "reme"}),
        ):
            meta = record("meta", thread=name)
            meta["payload"].update(fields)
            append(path.with_name(f"rollout-{name}.jsonl"), meta, record("UserMessage", "internal execution"))
        response = await CodexDialogueStep(**options(context, home))()
        assert response.success and response.answer["dialogue_messages"] == 0


@pytest.mark.asyncio
async def test_delivery_failure_reuses_frozen_batch_before_reading_further_changes(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "retry me"))
        original = delivery.side_effect
        delivery.side_effect = [{"deliveries": {}, "rejected": {}, "errors": {"work": "offline"}}]
        with pytest.raises(RuntimeError, match="delivery failed"):
            await CodexDialogueStep(**options(context, home))()
        frozen = delivery.await_args.kwargs["events"]
        append(path, record("UserMessage", "later", "user-2"))
        delivery.side_effect = original
        await CodexDialogueStep(**options(context, home, task_names=["proactive_notebook"]))()
        assert devices.call_count == 0 and delivery.await_args.kwargs["events"] == frozen
        assert delivery.await_args.kwargs["task_names"] == ["work"]
        assert (await inbox.snapshot("work"))["pending_count"] == 1
        assert [value["content"] for value in messages(context)] == ["retry me"]
        await CodexDialogueStep(**options(context, home))()
        assert sorted(value["content"] for value in messages(context)) == ["later", "retry me"]


@pytest.mark.asyncio
async def test_partial_delivery_retries_only_the_pending_recipient(tmp_path, source, monkeypatch):
    """A saved batch retains its recipients when configuration changes."""
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        agents = context.components["proactive_manager"]["default"]
        original = agents.append_events

        async def write(name, events):
            if name == "proactive_notebook":
                raise OSError("inbox unavailable")
            return await original(name, events)

        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "for both"))
        with monkeypatch.context() as patch:
            patch.setattr(agents, "append_events", write)
            with pytest.raises(RuntimeError, match="inbox unavailable"):
                await CodexDialogueStep(**options(context, home, task_names=["work", "proactive_notebook"]))()
        assert (await inbox.snapshot("work"))["pending_count"] == 1
        first = delivery.await_args.kwargs["events"]

        response = await CodexDialogueStep(**options(context, home, task_names=["work"]))()
        assert response.success
        assert delivery.await_args.kwargs == {
            "task_names": ["proactive_notebook"],
            "events": first,
        }
        assert (await inbox.snapshot("work"))["pending_count"] == 1
        assert (await inbox.snapshot("proactive_notebook"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_collectors_finish_a_batch_when_one_target_is_time_triggered(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path, work_cron="0 * * * *") as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        targets = ["work", "proactive_notebook"]
        for message in ("first", "later"):
            append(path, record("UserMessage", message, message))
            result = await CodexDialogueStep(**options(context, home, task_names=targets))()
            assert result.success and set(result.answer["rejected"]) == {"work"}
            assert not dialogue_index(context, home).pending_path.exists()
        assert (await inbox.snapshot("work"))["pending_count"] == 0
        assert (await inbox.snapshot("proactive_notebook"))["pending_count"] == 2

        screen = await capture_screen(context)
        assert screen.success
        receipt = (workspace(context) / screen.answer["path"]).with_name(".delivery.json")
        assert set(WorkspaceFiles(workspace(context)).read_json(receipt, {})["rejected"]) == {"work"}
        assert not (workspace(context) / "resource/.pending-resource-events.json").exists()
        assert devices.call_count == 1


@pytest.mark.asyncio
async def test_lost_delivery_response_does_not_duplicate_stream_events(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "once"))
        original = delivery.side_effect

        async def lost_response(**kwargs):
            await original(**kwargs)
            raise ConnectionError("lost reply")

        delivery.side_effect = lost_response
        with pytest.raises(ConnectionError):
            await CodexDialogueStep(**options(context, home))()
        delivery.side_effect = original
        await CodexDialogueStep(**options(context, home))()
        assert (await inbox.snapshot("work"))["pending_count"] == 1
        assert len(messages(context)) == 1


@pytest.mark.asyncio
async def test_partial_resource_publication_resumes_without_rescanning(tmp_path, source, devices, monkeypatch):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "recover"))
        write_json = WorkspaceFiles.write_json

        def fail_dialogue_once(store, target, value):
            if target.name == "metadata.json" and "codex_dialogue" in target.parts:
                raise OSError("publish interrupted")
            return write_json(store, target, value)

        monkeypatch.setattr(WorkspaceFiles, "write_json", fail_dialogue_once)
        with pytest.raises(OSError, match="interrupted"):
            await CodexDialogueStep(**options(context, home))()
        assert (await inbox.snapshot("work"))["pending_count"] == 0
        pending = progress(context).with_name("pending.json")
        batch = json.loads(pending.read_text())
        assert batch["resource"]["metadata"]["source"] == "codex_dialogue"
        assert batch["updates"] and batch["message_ids"]
        assert {path.name for path in pending.parent.iterdir() if path.is_file()} == {"progress.jsonl", "pending.json"}
        assert not list(workspace(context).glob("resource/*/*/*/metadata.json"))
        assert len(messages(context)) == 1
        monkeypatch.setattr(WorkspaceFiles, "write_json", write_json)
        monkeypatch.setattr(CodexDialogueIndex, "scan", Mock(side_effect=AssertionError("must retry the frozen batch")))
        await CodexDialogueStep(**options(context, home))()
        devices.assert_not_called()
        assert (await inbox.snapshot("work"))["pending_count"] == 1
        assert not pending.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_name", ["screenshot", "codex_dialogue"])
async def test_collection_never_overwrites_an_existing_resource_package(tmp_path, source, source_name, monkeypatch):
    home, path = source
    monkeypatch.setattr(
        "reme.steps.proactive.data_connectors.base_connector.uuid4",
        Mock(return_value=Mock(hex="a8f3c2" + "0" * 26)),
    )
    async with runtime(tmp_path) as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "keep existing resources"))
        directory = workspace(context) / "resource" / source_name / "2026-11-15/120000_a8f3c2"
        directory.mkdir(parents=True)
        original = directory / "original.txt"
        original.write_text("existing resource")
        before = progress(context).read_bytes()
        collector = (
            CodexDialogueStep(**options(context, home))
            if source_name == "codex_dialogue"
            else ScreenCaptureStep(app_context=context)
        )
        with pytest.raises(FileExistsError):
            await collector()
        assert original.read_text() == "existing resource"
        assert list(directory.iterdir()) == [original]
        assert progress(context).read_bytes() == before
        assert (await inbox.snapshot("work"))["pending_count"] == 0
        assert not (progress(context).with_name("pending.json")).exists()


@pytest.mark.asyncio
async def test_screenshot_failure_does_not_drop_dialogue(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "still delivered"))
        devices.side_effect = RuntimeError("screen permission denied")
        screen = await capture_screen(context)
        assert not screen.success and "screen permission denied" in screen.answer
        response = await CodexDialogueStep(**options(context, home))()
        assert response.success and response.answer["errors"] == {}
        assert devices.call_count == 1
        assert (await inbox.claim("work"))["items"][0]["source"] == "codex_dialogue"


@pytest.mark.asyncio
async def test_scanning_does_not_block_screens_and_cancellation_joins_its_worker(tmp_path, source, monkeypatch):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        entered, release = threading.Event(), threading.Event()
        thread_ids = []
        scan = CodexDialogueIndex.scan

        def slow_scan(index, *args):
            thread_ids.append(threading.get_ident())
            entered.set()
            assert release.wait(3)
            index.check_stopped()
            return scan(index, *args)

        monkeypatch.setattr(CodexDialogueIndex, "scan", slow_scan)
        step = CodexDialogueStep(**options(context, home))
        task = asyncio.create_task(step())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await asyncio.wait_for(asyncio.sleep(0.01), 0.2)
            assert thread_ids == [thread_ids[0]] and thread_ids[0] != threading.get_ident()
            screen = await asyncio.wait_for(capture_screen(context), 1)
            assert screen.success
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert step.cancelled.is_set()
        assert not list(workspace(context).glob("resource/codex_dialogue/*/*/metadata.json"))
        assert len(list(workspace(context).glob("resource/screenshot/*/*/metadata.json"))) == 1
        assert not (progress(context).with_name("pending.json")).exists()


@pytest.mark.asyncio
async def test_permissions_fail_before_source_or_device_access(tmp_path, source, devices):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        with pytest.raises(PermissionError):
            await CodexDialogueStep(**options(context, home))(RuntimeContext(_allowed_paths=[]))
        devices.assert_not_called()
        assert not progress(context).exists()


@pytest.mark.asyncio
async def test_checkpoint_failure_after_delivery_retries_the_same_events(tmp_path, source, monkeypatch, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "commit me"))
        commit = CodexDialogueIndex.commit

        def fail_commit(*_args):
            raise OSError("checkpoint unavailable")

        monkeypatch.setattr(CodexDialogueIndex, "commit", fail_commit)
        with pytest.raises(OSError, match="checkpoint unavailable"):
            await CodexDialogueStep(**options(context, home))()
        first = delivery.await_args.kwargs["events"]
        monkeypatch.setattr(CodexDialogueIndex, "commit", commit)
        await CodexDialogueStep(**options(context, home))()
        assert delivery.await_args.kwargs["events"] == first and devices.call_count == 0
        assert (await inbox.snapshot("work"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_interrupted_pending_cleanup_recovers_without_replaying_dialogue(tmp_path, source, devices, monkeypatch):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "commit once"))
        pending = progress(context).with_name("pending.json")
        unlink = type(pending).unlink

        def fail_cleanup(path, *args, **kwargs):
            if path == pending:
                raise OSError("cleanup interrupted")
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(type(pending), "unlink", fail_cleanup)
            with pytest.raises(OSError, match="cleanup interrupted"):
                await CodexDialogueStep(**options(context, home))()
        assert pending.is_file()
        committed = progress(context).read_bytes()
        assert len(messages(context)) == 1
        await CodexDialogueStep(**options(context, home))()
        assert len(messages(context)) == 1
        assert not pending.exists()
        assert progress(context).read_bytes() == committed
        assert devices.call_count == 0
        assert (await inbox.snapshot("work"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_screenshot_delivery_failure_does_not_block_dialogue_progress(
    tmp_path,
    source,
    devices,
    monkeypatch,
):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        proactive_runtime = context.components["proactive_manager"]["default"]
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "first", "first"))
        await CodexDialogueStep(**options(context, home))()
        committed = progress(context).read_bytes()
        append_events = proactive_runtime.append_events

        async def fail_screenshot(agent_name, events):
            if any(event.source == "screenshot" for event in events):
                raise ConnectionError("screen delivery failed")
            return await append_events(agent_name, events)

        with monkeypatch.context() as patch:
            patch.setattr(context.components["proactive_manager"]["default"], "append_events", fail_screenshot)
            screen = await capture_screen(context)
        assert not screen.success and "screen delivery failed" in screen.answer
        assert progress(context).read_bytes() == committed
        (journal,) = workspace(context).glob("resource/screenshot/*/*/.delivery.json")
        append(path, record("UserMessage", "later", "later"))
        result = await CodexDialogueStep(**options(context, home))()
        assert result.success and result.answer["dialogue_messages"] == 1
        assert sorted(message["content"] for message in messages(context)) == ["first", "later"]
        assert devices.call_count == 1 and delivery.await_count == 2
        assert (await inbox.snapshot("work"))["pending_count"] == 2
        devices.return_value = []
        devices.side_effect = None
        retry = await capture_screen(context, task_names=["proactive_notebook"])
        assert retry.success and retry.answer == {"skipped": "no_new_data"}
        assert (await inbox.snapshot("work"))["pending_count"] == 3
        assert (await inbox.snapshot("proactive_notebook"))["pending_count"] == 0
        assert WorkspaceFiles(workspace(context)).read_json(journal, {})["pending"] == []


@pytest.mark.asyncio
async def test_failed_dialogue_pending_write_preserves_screenshots_and_retries_dialogue(
    tmp_path,
    source,
    devices,
    monkeypatch,
):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        screen = await capture_screen(context)
        assert screen.success
        manifest = workspace(context) / screen.answer["path"]
        saved = manifest.read_bytes()
        append(path, record("UserMessage", "not lost"))
        pending = progress(context).with_name("pending.json")
        before = progress(context).read_bytes()
        write_json = WorkspaceFiles.write_json

        def fail_dialogue(store, target, value):
            if target == pending:
                raise OSError("dialogue persistence failed")
            return write_json(store, target, value)

        with monkeypatch.context() as patch:
            patch.setattr(WorkspaceFiles, "write_json", fail_dialogue)
            with pytest.raises(OSError, match="persistence failed"):
                await CodexDialogueStep(**options(context, home))()
        assert not pending.exists() and progress(context).read_bytes() == before
        assert manifest.read_bytes() == saved and not messages(context)
        await CodexDialogueStep(**options(context, home))()
        assert [event.source for event in delivery.await_args.kwargs["events"]] == ["codex_dialogue"]
        assert [message["content"] for message in messages(context)] == ["not lost"]
        await CodexDialogueStep(**options(context, home))()
        assert devices.call_count == 1 and delivery.await_count == 1
        assert (await inbox.snapshot("work"))["pending_count"] == 2


@pytest.mark.asyncio
async def test_corrupt_source_does_not_block_other_projects(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        bad = path.with_name("rollout-bad.jsonl")
        append(bad, record("meta", thread="bad"))
        with bad.open("ab") as handle:
            handle.write(b"{invalid json}\n")
        append(path, record("UserMessage", "valid project"))
        result = await CodexDialogueStep(**options(context, home))()
        assert not result.success and len(result.answer["errors"]) == 1
        assert [value["content"] for value in messages(context)] == ["valid project"]
        assert (await inbox.snapshot("work"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_invalid_cursor_and_changed_source_root_fail_closed(tmp_path, source, devices):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        with pytest.raises(ValueError, match="different source"):
            await CodexDialogueStep(**options(context, home.parent / "other"))()
        state = json.loads(progress(context).read_text())
        next(iter(state["files"].values()))["offset"] = -1
        progress(context).write_text(json.dumps(state) + "\n")
        with pytest.raises(ValueError, match="cursor"):
            await CodexDialogueStep(**options(context, home))()
        devices.assert_not_called()


@pytest.mark.asyncio
async def test_symlinked_sources_and_resource_roots_are_not_followed(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await CodexDialogueInitStep(**options(context, home))()
        outside = tmp_path / "outside.jsonl"
        append(outside, record("meta"), record("UserMessage", "outside"))
        path.with_name("rollout-link.jsonl").symlink_to(outside)
        await CodexDialogueStep(**options(context, home))()
        assert not messages(context) and (await inbox.snapshot("work"))["pending_count"] == 0
        devices.reset_mock()
        resources = workspace(context) / "resource"
        resources.rename(workspace(context) / "saved-resources")
        resources.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="symlinks"):
            await CodexDialogueStep(**options(context, home))()
        devices.assert_not_called()


@pytest.mark.asyncio
async def test_bounded_passes_rotate_between_files_and_finish_reconciliation(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await CodexDialogueInitStep(**options(context, home))()
        for number in range(3):
            append(path, record("UserMessage", "x" * 120, f"first-{number}"))
        other = path.with_name("rollout-other.jsonl")
        append(other, record("meta", thread="other"), record("UserMessage", "other", "other-1"))
        for _ in range(8):
            response = await CodexDialogueStep(**options(context, home, max_scan_bytes=500))()
            assert response.success
        assert len(messages(context)) == 4
        assert {value["message_id"] for value in messages(context)} == {
            "first-0",
            "first-1",
            "first-2",
            "other-1",
        }
        # Replacing a file requires multiple bounded passes. Previously imported IDs stay excluded in every pass.
        path.unlink()
        append(
            path,
            record("meta"),
            *(record("UserMessage", "x" * 120, f"first-{i}") for i in range(3)),
            record("UserMessage", "after reset", "first-3"),
        )
        for _ in range(8):
            await CodexDialogueStep(**options(context, home, max_scan_bytes=500))()
        assert len(messages(context)) == 5


@pytest.mark.asyncio
async def test_overlapping_collections_are_serialized(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await CodexDialogueInitStep(**options(context, home))()
        append(path, record("UserMessage", "one dialogue batch"))
        first, second = await asyncio.gather(
            CodexDialogueStep(**options(context, home))(),
            CodexDialogueStep(**options(context, home))(),
        )
        assert first.success and second.success and devices.call_count == 0
        assert [len(call.kwargs["events"]) for call in delivery.await_args_list] == [1]
        assert len(messages(context)) == 1 and (await inbox.snapshot("work"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_cron_establishes_baseline_before_first_wait_and_delivers_only_new_dialogue(tmp_path, source):
    """The configured init and periodic chains retain the first activation baseline."""
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        config = resolve_app_config(config="proactive", log_config=False)["jobs"]["codex_dialogue_cron"]
        for key in ("init_steps", "steps"):
            config[key][0]["codex_home"] = str(home)
        job = CronJob(name="codex_dialogue_cron", app_context=context, **config)
        waits = 0

        async def wait(_delay):
            nonlocal waits
            waits += 1
            if waits == 1:
                assert progress(context).is_file()
                assert not messages(context)
                delivery.assert_not_awaited()
                append(path, record("UserMessage", "new after startup", "new-1"))
            else:
                job._stop_event.set()

        job._next_fire_delay = lambda: 0
        job._wait_or_stop = wait
        await job.start()
        try:
            await asyncio.wait_for(job._task, 2)
        finally:
            await job.close()
        assert [message["message_id"] for message in messages(context)] == ["new-1"]
        delivery.assert_awaited_once()
        assert (await inbox.snapshot("work"))["pending_count"] == 1
        assert (await inbox.snapshot("proactive_notebook"))["pending_count"] == 1
