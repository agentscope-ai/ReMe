"""Tests for read-only job execution count evaluation helpers."""

import asyncio
from types import SimpleNamespace

import pytest

from reme.components.job import BaseJob, StreamJob
from reme.utils.evaluation_interface import check_job_count


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
