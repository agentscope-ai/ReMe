"""Incremental benchmark persistence and build-policy hooks without Docker."""

# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument

import pytest
from deploy.benchmark_docker import BenchmarkDockerPool, CasePlan, EvaluationQuery, JobRequest
from .test_benchmark_docker import FakeCase, FakeFactory


@pytest.mark.asyncio
async def test_pool_incremental_callback_and_partial_build_policy(tmp_path):
    events = []

    class PartialCase(FakeCase):
        async def run_build(self, jobs):
            await super().run_build(jobs)
            return {"success": False, "jobs": [{"result": {"success": False}}, {"result": {"success": True}}]}

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, **kwargs):
            return [PartialCase(name, self.events) for name in case_ids]

    completed = []

    async def finished(plan, result):
        completed.append(result)
        events.append((plan.case_id, "persist"))

    plans = [
        CasePlan(
            "case",
            build_jobs=(JobRequest("auto_memory"),),
            queries=(EvaluationQuery("q", "question", "answer"),),
        ),
    ]
    result = await BenchmarkDockerPool(Factory(events), concurrency=1).run(
        plans,
        on_case_finished=finished,
        accept_build=lambda build: len(build["jobs"]) == 2,
        fail_fast=True,
    )
    assert result == completed
    assert ("case", "query", 1) in events
    assert ("case", "persist") in events
    assert ("pool-0", "close") in events


@pytest.mark.asyncio
async def test_pool_callback_failure_still_closes_workers():
    events = []

    async def fail(plan, result):
        raise RuntimeError("persistence failed")

    with pytest.raises(RuntimeError, match="persistence failed"):
        await BenchmarkDockerPool(FakeFactory(events), concurrency=1).run(
            [CasePlan("case", build_jobs=(JobRequest("auto_memory"),))],
            on_case_finished=fail,
        )
    assert ("pool-0", "close") in events
