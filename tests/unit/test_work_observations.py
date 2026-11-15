"""Five-minute work observations use real event delivery and isolated Codex rollout files."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext, LocalAgentRuntime, R, RuntimeContext
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.agent_inbox import AgentEventInbox
from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum
from reme.schema import Response, ResourceMetadata
from reme.steps.proactive.data_connectors import ScreenCaptureStep, WorkObservationInitStep, WorkObservationStep
from reme.steps.proactive.data_connectors._codex_dialogue import CodexDialogueIndex
from reme.steps.proactive.data_connectors import work_observation

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
                "content": [{"type": "text", "text": text}],
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
    monkeypatch.setattr(work_observation, "datetime", clock)

    def capture(_step, directory):
        (directory / "1.png").write_bytes(b"fake-screen-1")
        (directory / "2.png").write_bytes(b"fake-screen-2")
        return [{"path": "1.png"}, {"path": "2.png"}]

    collector = Mock(side_effect=capture)

    def collect(step, directory):
        return collector(step, directory)

    monkeypatch.setattr(ScreenCaptureStep, "collect", collect)
    return collector


@asynccontextmanager
async def runtime(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path / "workspace"), language="zh")
    Path(context.app_config.workspace_dir).mkdir(exist_ok=True)
    inbox = AgentEventInbox(app_context=context)
    agents = LocalAgentRuntime(
        app_context=context,
        agents=[
            {
                "name": "work",
                "trigger_type": "event",
                "description": "跟踪项目进展和难点",
            },
        ],
    )
    context.components = {
        "agent_inbox": {"default": inbox},
        "agent_wrapper": {"default": Mock(spec=BaseAgentWrapper)},
        "agent_runtime": {"default": agents},
    }
    job = BaseJob(app_context=context, steps=[{"backend": "append_event_step"}])
    delivery = AsyncMock(side_effect=job.__call__)
    context.jobs["append_event"] = delivery
    async with inbox, agents, job:
        agents.watcher.cancel()
        await asyncio.gather(agents.watcher, return_exceptions=True)
        agents.watcher = None
        yield context, inbox, delivery


def options(context, home, **kwargs):
    return {"app_context": context, "codex_home": home, **kwargs}


def workspace(context):
    return Path(context.app_config.workspace_dir)


def checkpoint(context):
    return workspace(context) / "metadata/work_observations/work/state.json"


def messages(context):
    return [
        json.loads(line)
        for path in sorted(workspace(context).glob("resource/codex_dialogue/*/*/dialogue.jsonl"))
        for line in path.read_text().splitlines()
    ]


@pytest.mark.asyncio
async def test_startup_baseline_and_clean_dialogue_are_published_with_all_screens(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await WorkObservationInitStep(**options(context, home))()
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
        response = await WorkObservationStep(**options(context, home))()
        assert response.success and response.answer["dialogue_messages"] == 2
        assert response.answer["scanned_bytes"] == path.stat().st_size - original_size
        assert [message["content"] for message in messages(context)] == ["做增量感知", "开发完成"]
        assert {message["cwd"] for message in messages(context)} == {"/projects/ReMe2"}
        events = (await inbox.claim("work"))["items"]
        assert [event["source"] for event in events] == ["screenshot", "codex_dialogue"]
        assert len(events[0]["paths"]) == 3
        assert delivery.await_count == 1 and len(delivery.await_args.kwargs["events"]) == 2
        for event in events:
            manifest = workspace(context) / event["paths"][0]
            assert ResourceMetadata.model_validate_json(manifest.read_text()).source == event["source"]
            assert all(not Path(value).is_absolute() and "rollout" not in value for value in event["paths"])


@pytest.mark.asyncio
async def test_restart_catches_up_without_replaying_history_and_lost_index_is_rebuildable(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "first"))
        await WorkObservationStep(**options(context, home))()
    append(path, record("AgentMessage", "after restart", "final", phase="final_answer"))
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
        await WorkObservationStep(**options(context, home))()
        assert sorted(value["content"] for value in messages(context)) == ["after restart", "first"]
        checkpoint(context).unlink()
        await WorkObservationInitStep(**options(context, home))()
        response = await WorkObservationStep(**options(context, home))()
        assert response.answer["scanned_bytes"] == response.answer["dialogue_messages"] == 0


@pytest.mark.asyncio
async def test_tool_only_changes_advance_the_cursor(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("CommandExecution", "large output" * 100), record("Reasoning", "private"))
        first = await WorkObservationStep(**options(context, home))()
        second = await WorkObservationStep(**options(context, home))()
        assert first.answer["scanned_bytes"] > 0 and first.answer["dialogue_messages"] == 0
        assert second.answer["scanned_bytes"] == 0 and messages(context) == []


@pytest.mark.asyncio
async def test_incomplete_utf8_tail_is_retried_after_the_newline(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
        line = (json.dumps(record("UserMessage", "中文对话"), ensure_ascii=False) + "\n").encode()
        split = line.index("中文".encode()) + 1
        with path.open("ab") as handle:
            handle.write(line[:split])
        first = await WorkObservationStep(**options(context, home))()
        assert first.success and first.answer["dialogue_messages"] == 0
        with path.open("ab") as handle:
            handle.write(line[split:])
        second = await WorkObservationStep(**options(context, home))()
        assert second.success and [value["content"] for value in messages(context)] == ["中文对话"]


@pytest.mark.asyncio
async def test_new_files_in_old_directories_and_replaced_files_are_deduplicated(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "retained"))
        await WorkObservationStep(**options(context, home))()
        path.unlink()
        append(
            path,
            record("meta"),
            record("UserMessage", "retained"),
            record("UserMessage", "new after replacement", "user-2"),
        )
        other = path.with_name("rollout-other.jsonl")
        append(other, record("meta", thread="thread-2"), record("UserMessage", "other project", "user-3"))
        response = await WorkObservationStep(**options(context, home))()
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
        await WorkObservationInitStep(**options(context, home))()
        for name, fields in (
            ("subagent", {"source": {"subagent": {"thread_spawn": {}}}}),
            ("reme", {"originator": "reme"}),
        ):
            meta = record("meta", thread=name)
            meta["payload"].update(fields)
            append(path.with_name(f"rollout-{name}.jsonl"), meta, record("UserMessage", "internal execution"))
        response = await WorkObservationStep(**options(context, home))()
        assert response.success and response.answer["dialogue_messages"] == 0


@pytest.mark.asyncio
async def test_delivery_failure_reuses_frozen_batch_before_reading_further_changes(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "retry me"))
        original = delivery.side_effect
        delivery.side_effect = [Response(success=False, answer="offline")]
        with pytest.raises(RuntimeError, match="delivery failed"):
            await WorkObservationStep(**options(context, home))()
        frozen = delivery.await_args.kwargs["events"]
        append(path, record("UserMessage", "later", "user-2"))
        delivery.side_effect = original
        await WorkObservationStep(**options(context, home))()
        assert devices.call_count == 1 and delivery.await_args.kwargs["events"] == frozen
        assert (await inbox.snapshot("work"))["pending_count"] == 2
        assert [value["content"] for value in messages(context)] == ["retry me"]
        await WorkObservationStep(**options(context, home))()
        assert sorted(value["content"] for value in messages(context)) == ["later", "retry me"]


@pytest.mark.asyncio
async def test_lost_delivery_response_does_not_duplicate_stream_events(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "once"))
        original = delivery.side_effect

        async def lost_response(**kwargs):
            await original(**kwargs)
            raise ConnectionError("lost reply")

        delivery.side_effect = lost_response
        with pytest.raises(ConnectionError):
            await WorkObservationStep(**options(context, home))()
        delivery.side_effect = original
        await WorkObservationStep(**options(context, home))()
        assert (await inbox.snapshot("work"))["pending_count"] == 2
        assert len(messages(context)) == 1


@pytest.mark.asyncio
async def test_partial_resource_publication_resumes_without_recapture(tmp_path, source, devices, monkeypatch):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "recover"))
        rename = Path.rename

        def fail_dialogue_once(current, target):
            if current.name == "codex_dialogue":
                raise OSError("publish interrupted")
            return rename(current, target)

        monkeypatch.setattr(Path, "rename", fail_dialogue_once)
        with pytest.raises(OSError, match="interrupted"):
            await WorkObservationStep(**options(context, home))()
        assert (await inbox.snapshot("work"))["pending_count"] == 0
        monkeypatch.setattr(Path, "rename", rename)
        await WorkObservationStep(**options(context, home))()
        assert devices.call_count == 1 and (await inbox.snapshot("work"))["pending_count"] == 2


@pytest.mark.asyncio
async def test_screenshot_failure_does_not_drop_dialogue(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "still delivered"))
        devices.side_effect = RuntimeError("screen permission denied")
        response = await WorkObservationStep(**options(context, home))()
        assert not response.success and response.answer["errors"] == {"screenshot": "screen permission denied"}
        assert (await inbox.claim("work"))["items"][0]["source"] == "codex_dialogue"


@pytest.mark.asyncio
async def test_scanning_runs_off_loop_and_cancellation_joins_its_worker(tmp_path, source, monkeypatch):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
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
        step = WorkObservationStep(**options(context, home))
        task = asyncio.create_task(step())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await asyncio.wait_for(asyncio.sleep(0.01), 0.2)
            assert thread_ids == [thread_ids[0]] and thread_ids[0] != threading.get_ident()
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert step.cancelled.is_set()
        assert not list(workspace(context).glob("resource/*/*/*/metadata.json"))
        assert not (checkpoint(context).parent / "pending").exists()


@pytest.mark.asyncio
async def test_permissions_fail_before_source_or_device_access(tmp_path, source, devices):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        with pytest.raises(PermissionError):
            await WorkObservationStep(**options(context, home))(RuntimeContext(_allowed_paths=[]))
        devices.assert_not_called()
        assert not checkpoint(context).exists()


@pytest.mark.asyncio
async def test_checkpoint_failure_after_delivery_retries_the_same_events(tmp_path, source, monkeypatch, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "commit me"))
        commit = CodexDialogueIndex.commit

        def fail_commit(*_args):
            raise OSError("checkpoint unavailable")

        monkeypatch.setattr(CodexDialogueIndex, "commit", fail_commit)
        with pytest.raises(OSError, match="checkpoint unavailable"):
            await WorkObservationStep(**options(context, home))()
        first = delivery.await_args.kwargs["events"]
        monkeypatch.setattr(CodexDialogueIndex, "commit", commit)
        await WorkObservationStep(**options(context, home))()
        assert delivery.await_args.kwargs["events"] == first and devices.call_count == 1
        assert (await inbox.snapshot("work"))["pending_count"] == 2


@pytest.mark.asyncio
async def test_interrupted_pending_cleanup_recovers_without_replaying_dialogue(tmp_path, source, monkeypatch):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "commit once"))
        pending = checkpoint(context).parent / "pending"
        rmdir = type(pending).rmdir

        def fail_cleanup(directory):
            if directory == pending:
                raise OSError("cleanup interrupted")
            return rmdir(directory)

        with monkeypatch.context() as patch:
            patch.setattr(type(pending), "rmdir", fail_cleanup)
            with pytest.raises(OSError, match="cleanup interrupted"):
                await WorkObservationStep(**options(context, home))()
        assert pending.is_dir() and not list(pending.iterdir())
        assert len(messages(context)) == 1
        await WorkObservationStep(**options(context, home))()
        assert len(messages(context)) == 1
        assert not pending.exists()
        assert (await inbox.snapshot("work"))["pending_count"] == 3


@pytest.mark.asyncio
async def test_corrupt_source_does_not_block_other_projects(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await WorkObservationInitStep(**options(context, home))()
        bad = path.with_name("rollout-bad.jsonl")
        append(bad, record("meta", thread="bad"))
        with bad.open("ab") as handle:
            handle.write(b"{invalid json}\n")
        append(path, record("UserMessage", "valid project"))
        result = await WorkObservationStep(**options(context, home))()
        assert not result.success and len(result.answer["errors"]) == 1
        assert [value["content"] for value in messages(context)] == ["valid project"]
        assert (await inbox.snapshot("work"))["pending_count"] == 2


@pytest.mark.asyncio
async def test_invalid_cursor_and_changed_source_root_fail_closed(tmp_path, source, devices):
    home, _ = source
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
        with pytest.raises(ValueError, match="different source"):
            await WorkObservationStep(**options(context, home.parent / "other"))()
        state = json.loads(checkpoint(context).read_text())
        next(iter(state["files"].values()))["offset"] = -1
        checkpoint(context).write_text(json.dumps(state))
        with pytest.raises(ValueError, match="cursor"):
            await WorkObservationStep(**options(context, home))()
        devices.assert_not_called()


@pytest.mark.asyncio
async def test_symlinked_sources_and_resource_roots_are_not_followed(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, _):
        await WorkObservationInitStep(**options(context, home))()
        outside = tmp_path / "outside.jsonl"
        append(outside, record("meta"), record("UserMessage", "outside"))
        path.with_name("rollout-link.jsonl").symlink_to(outside)
        await WorkObservationStep(**options(context, home))()
        assert not messages(context) and (await inbox.snapshot("work"))["pending_count"] == 1
        devices.reset_mock()
        resources = workspace(context) / "resource"
        resources.rename(workspace(context) / "saved-resources")
        resources.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="symlinks"):
            await WorkObservationStep(**options(context, home))()
        devices.assert_not_called()


@pytest.mark.asyncio
async def test_bounded_passes_rotate_between_files_and_finish_reconciliation(tmp_path, source):
    home, path = source
    async with runtime(tmp_path) as (context, _, _):
        await WorkObservationInitStep(**options(context, home))()
        for number in range(3):
            append(path, record("UserMessage", "x" * 120, f"first-{number}"))
        other = path.with_name("rollout-other.jsonl")
        append(other, record("meta", thread="other"), record("UserMessage", "other", "other-1"))
        for _ in range(8):
            response = await WorkObservationStep(**options(context, home, max_scan_bytes=500))()
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
            await WorkObservationStep(**options(context, home, max_scan_bytes=500))()
        assert len(messages(context)) == 5


@pytest.mark.asyncio
async def test_overlapping_collections_are_serialized(tmp_path, source, devices):
    home, path = source
    async with runtime(tmp_path) as (context, inbox, delivery):
        await WorkObservationInitStep(**options(context, home))()
        append(path, record("UserMessage", "one dialogue batch"))
        first, second = await asyncio.gather(
            WorkObservationStep(**options(context, home))(),
            WorkObservationStep(**options(context, home))(),
        )
        assert first.success and second.success and devices.call_count == 2
        assert [len(call.kwargs["events"]) for call in delivery.await_args_list] == [2, 1]
        assert len(messages(context)) == 1 and (await inbox.snapshot("work"))["pending_count"] == 3


def test_config_initializes_on_startup_and_collects_both_sources_every_five_minutes(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    config = resolve_app_config(config="proactive", log_config=False)
    jobs = config["jobs"]
    assert jobs["work_observation_init"]["backend"] == "background"
    assert jobs["work_observation_cron"]["cron"] == "*/5 * * * *"
    assert jobs["screen_capture_cron"]["cron"] == "*/5 * * * *"
    assert not jobs["screen_capture_cron"]["enabled"]
    for name, cls in (
        ("work_observation_init", WorkObservationInitStep),
        ("work_observation_cron", WorkObservationStep),
    ):
        spec = jobs[name]["steps"][0]
        assert spec["codex_home"] == "~/.codex"
        assert R.get(ComponentEnum.STEP, spec["backend"]) is cls
