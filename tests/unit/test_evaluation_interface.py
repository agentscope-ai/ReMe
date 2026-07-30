"""Tests for read-only job execution count evaluation helpers."""

import asyncio
from types import SimpleNamespace

import pytest

from reme.components.job import BaseJob, StreamJob
from reme.utils import global_counter_add
from reme.utils.evaluation_interface import (
    check_agent_token_count,
    check_agent_token_usage,
    check_job_count,
    track_agent_token_usage,
    track_agent_token_counts,
    track_job_counts,
)


def test_check_job_count_reads_registered_base_job_count():
    """check_job_count returns the number of completed BaseJob invocations."""

    async def run():
        app_context = SimpleNamespace(metadata={}, jobs={})
        job = BaseJob(name="search", app_context=app_context)
        app_context.jobs[job.name] = job

        await job()
        await job()

        assert check_job_count("search", app_context) == 2

    asyncio.run(run())


def test_check_job_count_resolves_custom_job_inheritance_path():
    """check_job_count resolves counters recorded under subclassed job paths."""

    async def run():
        class ProjectStreamJob(StreamJob):
            """Project-specific StreamJob subclass used to exercise MRO lookup."""

        app_context = SimpleNamespace(metadata={}, jobs={})
        job = ProjectStreamJob(name="chat", app_context=app_context)
        app_context.jobs[job.name] = job

        await job(stream_queue=asyncio.Queue())

        assert check_job_count("chat", app_context) == 1

    asyncio.run(run())


def test_check_job_count_rejects_unknown_job_name():
    """Unknown job names raise KeyError, matching Application.run_job."""
    app_context = SimpleNamespace(metadata={}, jobs={})

    with pytest.raises(KeyError, match="Job 'missing' not found"):
        check_job_count("missing", app_context)


def test_track_job_counts_returns_calls_made_inside_context():
    """The context manager reports only the calls made in its body."""

    async def run():
        app_context = SimpleNamespace(metadata={}, jobs={})
        search = BaseJob(name="search", app_context=app_context)
        app_context.jobs[search.name] = search

        await search()
        with track_job_counts(["search"], app_context) as counts:
            await search()
            await search()

        assert counts == {"search": 2}

    asyncio.run(run())


def test_track_job_counts_updates_results_when_body_raises():
    """Calls made before an exception are still included in the delta."""

    async def run():
        app_context = SimpleNamespace(metadata={}, jobs={})
        search = BaseJob(name="search", app_context=app_context)
        app_context.jobs[search.name] = search
        counts = {}

        with pytest.raises(RuntimeError, match="boom"):
            with track_job_counts(["search"], app_context) as counts:
                await search()
                raise RuntimeError("boom")

        assert counts == {"search": 1}

    asyncio.run(run())


def test_track_agent_token_counts_returns_delta_for_one_agent():
    """Token tracking mirrors job-count tracking over the token counter tree."""
    app_context = SimpleNamespace(metadata={}, jobs={})
    global_counter_add(app_context.metadata, ["__token_counter", "bench", "total_tokens"], 10)

    with track_agent_token_counts(["bench"], app_context) as counts:
        global_counter_add(app_context.metadata, ["__token_counter", "bench", "total_tokens"], 25)

    assert counts == {"bench": 25}
    assert check_agent_token_count("bench", app_context) == 35


def test_track_agent_token_usage_reports_only_supported_metrics():
    """Detailed usage tracking reports the shared input/output contract."""
    app_context = SimpleNamespace(metadata={}, jobs={})
    for metric, value in (("input_tokens", 10), ("output_tokens", 5), ("total_tokens", 15)):
        global_counter_add(app_context.metadata, ["__token_counter", "bench", metric], value)

    with track_agent_token_usage(["bench"], app_context) as usages:
        for metric, value in (("input_tokens", 20), ("output_tokens", 7), ("total_tokens", 27)):
            global_counter_add(app_context.metadata, ["__token_counter", "bench", metric], value)

    assert usages == {
        "bench": {
            "input_tokens": 20,
            "output_tokens": 7,
            "total_tokens": 27,
        },
    }


def test_track_agent_token_usage_keeps_unavailable_usage_as_none():
    """A backend that reports no usage remains unavailable to benchmarks."""
    app_context = SimpleNamespace(metadata={}, jobs={})

    with track_agent_token_usage(["bench"], app_context) as usages:
        pass

    assert usages == {
        "bench": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    }
