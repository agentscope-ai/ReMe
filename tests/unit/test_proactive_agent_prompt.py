"""Proactive prompt rendering across languages and trigger types."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import datetime, timedelta
import re

import pytest

from reme.steps.proactive.agent import ProactiveAgentStep
from .test_agent_runtime import create_agent, running_runtime


@pytest.mark.parametrize(
    "timestamp,yesterday",
    [
        ("2026-01-01T00:00:00+08:00", "2025-12-31"),
        ("2024-03-01T00:00:00+08:00", "2024-02-29"),
    ],
)
@pytest.mark.asyncio
async def test_round_paths_keep_both_calendar_summaries_and_paired_filenames(tmp_path, timestamp, yesterday):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        agent = await runtime.definition("monitor")
        step = ProactiveAgentStep(app_context=context)
        now = datetime.fromisoformat(timestamp)
        paths = step._paths(agent, now)
        assert paths["summary_path"] == tmp_path / "daily" / f"{now:%Y-%m-%d}" / "monitor.md"
        assert paths["yesterday_summary_path"] == tmp_path / "daily" / yesterday / "monitor.md"
        assert paths["detail_path"] == tmp_path / "daily" / f"{now:%Y-%m-%d}" / f"monitor_{now:%H%M%S}.md"
        assert paths["reminder_path"] == (
            tmp_path / "reme_agents/monitor/reminder" / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}.md"
        )
        assert not paths["summary_path"].exists() and not paths["yesterday_summary_path"].exists()
        for delta in (-5, -2, 1):
            path = tmp_path / "daily" / f"{now + timedelta(days=delta):%Y-%m-%d}" / "monitor.md"
            runtime.files.write(path, "# Summary\n")
        runtime.files.write(tmp_path / "daily/not-a-date/monitor.md", "# Invalid date\n")
        paths = step._paths(agent, now)
        assert paths["yesterday_summary_path"] == tmp_path / "daily" / yesterday / "monitor.md"
        assert not paths["yesterday_summary_path"].exists()
        runtime.files.write(paths["yesterday_summary_path"], "# Yesterday\n")
        runtime.files.write(paths["summary_path"], "# Today\n")
        runtime.files.write(paths["detail_path"], "# Existing details\n")
        paths = step._paths(agent, now)
        assert paths["summary_path"].read_text() == "# Today\n"
        assert paths["yesterday_summary_path"].read_text() == "# Yesterday\n"
        assert paths["detail_path"].name == f"monitor_{now:%H%M%S}_1.md"
        assert paths["reminder_path"].name == f"{now:%H%M%S}_1.md"
        runtime.files.write(paths["reminder_path"], "---\npriority: P1\n---\nExisting reminder\n")
        paths = step._paths(agent, now)
        assert paths["detail_path"].name == f"monitor_{now:%H%M%S}_2.md"
        assert paths["reminder_path"].name == f"{now:%H%M%S}_2.md"


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("trigger_type", ["cron", "event"])
@pytest.mark.parametrize("has_today_summary", [False, True])
@pytest.mark.asyncio
async def test_round_query_renders_path_aliases_and_trigger_specific_inputs(
    tmp_path,
    language,
    trigger_type,
    has_today_summary,
):
    async with running_runtime(tmp_path, language=language) as (runtime, context, _):
        values = {"name": "monitor", "description": "Monitor", "trigger_type": trigger_type}
        if trigger_type == "cron":
            values["cron"] = "0 * * * *"
        await runtime.create(values)
        agent = await runtime.definition("monitor")
        step = ProactiveAgentStep(app_context=context)
        now = runtime.now()
        paths = step._paths(agent, now)
        if has_today_summary:
            runtime.files.write(paths["summary_path"], "# Today\n")
        runtime.files.write(paths["yesterday_summary_path"], "# Yesterday\n")
        events = [
            {
                "event_ts": (now - timedelta(minutes=number)).isoformat(),
                "source": "test_event_source",
                "description": f"Observation {number}",
                "paths": [f"resource/observation_{number}.md", f"resource/screenshot_{number}.png"],
            }
            for number in (2, 1)
        ]
        query = step._query(agent, events, paths, now)
        for alias, key in (
            ("TASK_MEMORY", "memory_path"),
            ("DAILY_DIR", "daily_dir"),
            ("TODAY_SUMMARY", "summary_path"),
            ("YESTERDAY_SUMMARY", "yesterday_summary_path"),
            ("CURRENT_RECORD", "detail_path"),
            ("CURRENT_REMINDER", "reminder_path"),
        ):
            assert f"`{alias}`" in query
            assert query.count(f"`{paths[key]}`") == 1
        assert re.findall(r"^(\d+)\. ", query, re.MULTILINE) == [str(number) for number in range(1, 9)]
        summary_read = re.search(r"^2\. (.*)$", query, re.MULTILINE).group(1)
        assert "`YESTERDAY_SUMMARY`" in summary_read
        assert ("`TODAY_SUMMARY`" in summary_read) is has_today_summary
        assert not re.search(r"\{[a-z_]+\}|\[(?:cron|event|new_day|same_day)\]", query)
        for event in events:
            assert event["event_ts"] in query
            if trigger_type == "event":
                assert event["source"] in query and event["description"] in query
                for path in event["paths"]:
                    assert f"`{tmp_path / path}`" in query
            else:
                assert event["source"] not in query and event["description"] not in query
        assert ("web_search" in query) is (trigger_type == "cron")
        assert ("web_extract" in query) is (trigger_type == "cron")
