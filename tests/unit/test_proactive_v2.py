"""One daily record carries the complete v2 decision and delivery state."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import datetime
import os
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.config import resolve_app_config
from reme.core.registry import R
from reme.enumeration import ComponentEnum
from reme.steps.proactive.agent import ProactiveAgentStep, ProactiveAgentStepV2
from .test_agent_runtime import completed_result, create_agent, running_runtime
from .test_proactive_run import create_round_task

EMPTY = "---\nhas_content: false\nremind: false\nreminder_priority: null\nreminder_message: null\n---\n"
REMIND = (
    "---\nhas_content: {has_content}\nremind: true\n"
    "reminder_priority: {priority}\nreminder_message: Verified result\n---{body}"
)


def install_output(runtime, wrapper, monkeypatch, path, text):
    queries = []

    async def reply(query, **options):
        queries.append(query)
        runtime.files.write(path, text)
        async for chunk in type(wrapper).reply_stream(wrapper, query, **options):
            yield chunk

    monkeypatch.setattr(wrapper, "reply_stream", reply)
    return queries


def test_v2_profile_selects_one_backend_for_both_task_types():
    v1 = resolve_app_config(config="proactive", log_config=False)["jobs"]["run_proactive_agent"]
    v2 = resolve_app_config(config="proactive_v2", log_config=False)["jobs"]["run_proactive_agent"]
    assert R.get(ComponentEnum.STEP, v1["steps"][0]["backend"]) is ProactiveAgentStep
    assert R.get(ComponentEnum.STEP, v2["steps"][0]["backend"]) is ProactiveAgentStepV2
    assert v1.keys() == v2.keys()
    assert v1["enable_serve"] is False and v2["enable_serve"] is False


@pytest.mark.parametrize(
    "agent_type,language,priority,body",
    [("proactive", "zh", "P1", "\n\n    indented code\n\n"), ("cron", "en", "P0", "\n")],
)
@pytest.mark.asyncio
async def test_v2_round_publishes_and_acknowledges_the_same_note(
    tmp_path,
    monkeypatch,
    agent_type,
    language,
    priority,
    body,
):
    async with running_runtime(tmp_path, language=language) as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, agent_type)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStepV2(app_context=context)
        path = step._paths(await runtime.definition("monitor"), now)["record_path"]
        other_output = REMIND.format(has_content=str(bool(body.strip())).lower(), priority=priority, body=body)
        queries = install_output(runtime, wrapper, monkeypatch, path, other_output)

        result = await step(agent_name="monitor")
        assert result.success and result.answer["processed"] == 1
        assert len(queries) == 1 and "has_content: false" in queries[0]
        assert list((tmp_path / "daily").rglob("*.md")) == [path]
        assert not list((tmp_path / "reme_agents").glob("*/summary"))
        assert not list((tmp_path / "reme_agents").glob("*/reminder"))

        await runtime.outbox.sync()
        (ref,) = runtime.outbox.committed
        related = [path.relative_to(tmp_path).as_posix()]
        (item,) = (await runtime.notifications.read())["items"]
        assert item["message"] == "Verified result"
        assert item["priority"] == priority
        assert runtime.outbox.supporting_paths(item) == related == item["paths"]
        await runtime.notifications.consume([ref.id])
        await runtime.outbox.read("monitor", ref.id, mark_read=True)
        saved = path.read_text()
        assert saved.endswith(f"---{body}")
        state = frontmatter.loads(saved)["reminder_state"]
        assert all(state[key] for key in ("delivered_at", "read_at", "user_read_at"))

    async with running_runtime(tmp_path) as (runtime, _, _):
        item = await runtime.outbox.read("monitor", ref.id)
        assert item["paths"] == related and item["read_at"] and item["user_read_at"]


@pytest.mark.asyncio
async def test_recent_history_uses_round_order_and_injects_only_own_paths(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = datetime.fromisoformat("2026-11-20T11:00:00+08:00")
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStepV2(app_context=context, max_recent_records=3)
        paths = step._paths(await runtime.definition("monitor"), now)

        daily = paths["daily_dir"]
        yesterday = daily / "2026-11-19"
        today = daily / "2026-11-20"
        sources = [
            (yesterday / "monitor_235959.md", "---\naction: write\n---\nprivate old body"),
            (today / "monitor_090000_2.md", "private old body"),
            (today / "monitor_090000_10.md", REMIND.format(has_content="false", priority="P0", body="\n")),
            (today / "other_100000.md", "other task"),
            (today / "monitor_100001.md", EMPTY),
            (today / "monitor_100002.md", "---\naction: skip\n---\n"),
            (yesterday / "monitor_230000.md", "older than the selection"),
        ]
        for file, content in sources:
            runtime.files.write(file, content)
        os.utime(sources[0][0], (now.timestamp(), now.timestamp()))

        queries = install_output(runtime, wrapper, monkeypatch, paths["record_path"], EMPTY)
        assert (await step(agent_name="monitor")).success
        (query,) = queries
        recent_heading = query.index("\n## Recent records for this task\n")
        steps_heading = query.index("\n# Steps\n")
        assert query.index("Monitor experiments") < query.index("Finished") < recent_heading < steps_heading
        recent_section = query[recent_heading:steps_heading]
        assert (
            recent_section.index("monitor_090000_10.md")
            < recent_section.index("monitor_090000_2.md")
            < recent_section.index("monitor_235959.md")
        )
        assert all(file.name not in query for file, _ in sources[3:])
        assert "private old body" not in query and "Verified result" not in query
        assert not (await runtime.outbox.list("monitor"))["items"]


@pytest.mark.parametrize(
    "draft,error",
    [
        (EMPTY + "Unexpected body", ValueError),
        (EMPTY.replace("reminder_priority: null", "reminder_priority: P1"), ValueError),
        (EMPTY.replace("has_content: false", 'has_content: "false"'), ValueError),
        (REMIND.format(has_content="false", priority="P1", body="\n"), ValueError),
        (REMIND.format(has_content="true", priority="P1", body="\nNew result\n"), OSError),
    ],
)
@pytest.mark.asyncio
async def test_failed_v2_round_keeps_the_input_and_removes_only_its_draft(tmp_path, monkeypatch, draft, error):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStepV2(app_context=context)
        path = step._paths(await runtime.definition("monitor"), now)["record_path"]
        install_output(runtime, wrapper, monkeypatch, path, draft)
        if error is OSError:
            monkeypatch.setattr(runtime.inbox, "ack", AsyncMock(side_effect=OSError("receipt unavailable")))
        with pytest.raises(error):
            await step(agent_name="monitor")
        assert not path.exists()
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.asyncio
async def test_shared_outbox_retains_a_v1_reminder_after_a_v2_round(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.append(
            "monitor",
            {"id": "event-2", "event_ts": runtime.now(), "source": "experiment", "description": "New result"},
        )
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStepV2(app_context=context)
        path = step._paths(await runtime.definition("monitor"), now)["record_path"]
        draft = REMIND.format(has_content="true", priority="P1", body="\nNew result\n")
        install_output(runtime, wrapper, monkeypatch, path, draft)
        assert (await step(agent_name="monitor")).success
        await runtime.outbox.sync()

    async with running_runtime(tmp_path) as (runtime, _, _):
        items = (await runtime.outbox.list("monitor"))["items"]
        assert {item["priority"] for item in items} == {"P0", "P1"}
