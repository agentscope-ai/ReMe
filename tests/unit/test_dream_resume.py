"""Exercise Dream recovery with fresh Applications and a controlled local provider."""

# pylint: disable=protected-access

import asyncio
import copy
import json
import os
from contextlib import asynccontextmanager, nullcontext
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agentscope.model import ChatModelBase

from reme.application import Application
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.config import resolve_app_config
from reme.steps.evolve.dream.checkpoint import CheckpointStore, DreamCheckpoint, source_versions
from reme.steps.evolve.dream.finish import DreamFinishStep
from reme.steps.evolve.dream.integrate import DreamIntegrateStep

DAY = "2026-10-23"
SOURCE = f"daily/{DAY}/source.md"
TARGETS = ["digest/wiki/first.md", "digest/wiki/second.md"]
UNITS = [
    {"name": name, "bucket": "wiki", "summary": f"{name} memory", "paths": [SOURCE]} for name in ("first", "second")
]


class DreamCase:
    """Own isolated files, fresh application lifecycles, and provider call observations."""

    def __init__(self, workspace):
        self.workspace, self.metadata = workspace, "metadata"
        self.source = workspace / SOURCE
        self.source.parent.mkdir(parents=True)
        self.source.write_text("Original source", encoding="utf-8")
        self.units, self.calls, self.prompts = copy.deepcopy(UNITS), [], []

    @property
    def checkpoints(self):
        """Return only committed JSON checkpoints, excluding atomic-write temporary files."""
        return list((self.workspace / self.metadata / "dream").glob("*.json"))

    @property
    def saved(self):
        """Read progress from disk, without borrowing any application state."""
        assert len(self.checkpoints) == 1
        return DreamCheckpoint.model_validate_json(self.checkpoints[0].read_text())

    @asynccontextmanager
    async def open(self):
        """Start real local catalogs and jobs without external services or model providers."""
        defaults = resolve_app_config(config="default", log_config=False)
        types = ("tokenizer", "keyword_index", "file_graph", "tag_index", "file_store", "file_catalog")
        app = Application(
            workspace_dir=str(self.workspace),
            metadata_dir=self.metadata,
            enable_logo=False,
            log_to_file=False,
            log_to_console=False,
            service={"backend": "http", "web_enabled": False, "mcp_enabled": False},
            components={name: copy.deepcopy(defaults["components"][name]) for name in types},
            jobs={name: copy.deepcopy(defaults["jobs"][name]) for name in ("auto_dream", "write")},
        )
        self.calls.clear()
        self.prompts.clear()
        await app.start()
        try:
            yield app
        finally:
            await app.close()

    async def write(self, app, index, content):
        """Exercise the normal workspace-contained write job."""
        response = await app.run_job("write", path=TARGETS[index], content=content)
        assert response.success, response.answer

    @staticmethod
    def receipt(index, action):
        """Return a valid integration receipt for a controlled target."""
        return {"result": json.dumps({"action": action, "target_path": TARGETS[index]})}

    async def run(self, app, *, reply=None, interrupt="", fail=False, **options):
        """Track extraction and integration independently; custom replies control only integration."""

        async def provider(message, **kwargs):
            if "list_tags" in kwargs["job_tools"]:
                return {"result": "No tags added"}
            if kwargs["job_tools"] == ["read"]:
                self.calls.append("extract")
                return {"result": json.dumps({"units": self.units})}
            text = str(message)
            index = 0 if "first memory" in text else 1
            self.calls.append(UNITS[index]["name"])
            self.prompts.append(text)
            if reply:
                return await reply(index, text)
            if index == 1 and interrupt == "before_write":
                raise asyncio.CancelledError
            if index == 1 and fail:
                return {"result": "{}"}
            existed = (self.workspace / TARGETS[index]).exists()
            if not existed:
                await self.write(app, index, UNITS[index]["summary"])
            if index == 1 and interrupt == "after_write":
                raise asyncio.CancelledError
            return self.receipt(index, "CORROBORATE" if existed else "CREATE")

        agent = AsyncMock(spec=BaseAgentWrapper)
        agent.reply.side_effect = provider
        return await app.run_job(
            "auto_dream",
            date=DAY,
            scan_days=1,
            agent_wrapper=agent,
            as_llm=Mock(spec=ChatModelBase),
            **options,
        )

    async def interrupt(self, stage):
        """Leave disk state at a named interruption boundary, then discard the Application."""
        boundary = {
            "after_extract": (DreamIntegrateStep, "execute"),
            "before_finish": (DreamFinishStep, "execute"),
            "after_commit": (CheckpointStore, "remove"),
        }.get(stage)
        async with self.open() as app:
            with (
                patch.object(*boundary, side_effect=asyncio.CancelledError) if boundary else nullcontext(),
                pytest.raises(asyncio.CancelledError),
            ):
                await self.run(app, interrupt=stage)
            if stage != "after_commit":
                assert await app.context.components["file_catalog"]["dream"].get_nodes([SOURCE]) == []


@pytest.fixture(name="case")
def dream_case(tmp_path):
    """Each case starts with user-owned source material in a temporary workspace."""
    return DreamCase(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["after_extract", "before_write", "after_write", "before_finish", "no_llm"])
async def test_restart_reuses_extraction_and_completed_units(case, stage):
    """Resume only pending units; completed work and source content stay untouched."""
    original = case.source.read_bytes()
    await case.interrupt("before_finish" if stage == "no_llm" else stage)
    saved, logs = case.saved, []
    assert saved.state.units == UNITS
    assert saved.completed == ([] if stage == "after_extract" else [0] if "write" in stage else [0, 1])
    if "write" in stage:
        assert saved.active == 1
    restore = CheckpointStore.restore

    def capture(store, sources):
        store.logger = Mock(wraps=store.logger)
        state = restore(store, sources)
        logs.extend(call.args[0] for call in store.logger.info.call_args_list)
        return state

    async with case.open() as app:
        with (
            patch.object(CheckpointStore, "restore", capture),
            (
                patch("reme.steps.evolve.dream.integrate.llm_available", return_value=False)
                if stage == "no_llm"
                else nullcontext()
            ),
        ):
            response = await case.run(app)
        assert response.success, response.answer
        assert case.calls == (
            ["first", "second"] if stage == "after_extract" else ["second"] if "write" in stage else []
        )
        assert len(response.metadata["dream"]["integrate_results"]) == 2
        assert SOURCE in response.metadata["dream"]["checkpoint_paths"]
        assert any(f"resuming: {len(saved.completed)} of 2 units completed" in line for line in logs)
        assert not case.checkpoints
        assert case.source.read_bytes() == original
        for index, path in enumerate(TARGETS):
            assert (case.workspace / path).read_text().strip() == UNITS[index]["summary"]
        if stage == "after_write":
            assert TARGETS[1] in case.prompts[0] and "do not append duplicate evidence" in case.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["edit", "same_mtime", "add", "delete", "hint", "max_units", "corrupt", "version", "progress"],
)
async def test_incompatible_checkpoint_reextracts(case, change):
    """Never silently reuse extraction for changed inputs or invalid saved progress."""
    await case.interrupt("after_extract")
    options, path = {}, case.checkpoints[0]
    if change in ("edit", "same_mtime"):
        stat = case.source.stat()
        case.source.write_text("Modified source")
        if change == "same_mtime":
            os.utime(case.source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "add":
        (case.source.parent / "another.md").write_text("New source")
    elif change == "delete":
        case.source.unlink()
    elif change in ("hint", "max_units"):
        options[change] = "New guidance" if change == "hint" else 1
    elif change == "corrupt":
        path.write_text("{incomplete JSON")
    else:
        data = json.loads(path.read_text())
        data["version" if change == "version" else "completed"] = 999 if change == "version" else [0]
        path.write_text(json.dumps(data))
    case.units = []
    async with case.open() as app:
        response = await case.run(app, **options)
        assert response.success, response.answer
        assert case.calls == ["extract"]


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_restart_after_catalog_commit_only_cleans_up(case, empty):
    """A durable final watermark needs no new model work, including empty extraction."""
    case.metadata = "custom-metadata"
    if empty:
        case.units = []
    await case.interrupt("after_commit")
    assert case.checkpoints and not (case.workspace / "metadata").exists()
    async with case.open() as app:
        response = await case.run(app)
        assert response.success, response.answer
        assert not case.calls and not case.checkpoints


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", ["shared", "independent", "missing_target"])
async def test_partial_failure_retries_only_pending_work(case, sources):
    """Independent source watermarks cannot hide saved units needing target repair."""
    if sources != "shared":
        other = f"daily/{DAY}/other.md"
        (case.workspace / other).write_text("Another source")
        case.units[1]["paths"] = [other]
    async with case.open() as app:
        response = await case.run(app, fail=True)
        assert not response.success
        dream = response.metadata["dream"]
        assert (SOURCE in dream["checkpoint_paths"]) is (sources != "shared")
        assert case.units[1]["paths"][0] not in dream["checkpoint_paths"]
    if sources == "missing_target":
        (case.workspace / TARGETS[0]).unlink()
    async with case.open() as app:
        response = await case.run(app)
        assert response.success, response.answer
        assert case.calls == (["first", "second"] if sources == "missing_target" else ["second"])
        assert len(response.metadata["dream"]["integrate_results"]) == 2
        assert not response.metadata["dream"]["failed_units"] and not response.metadata["dream"]["errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_receipt", ["invalid", "exception"])
@pytest.mark.parametrize("stage", ["after_extract", "before_write"])
async def test_resume_requires_receipt_despite_partial_writes(case, bad_receipt, stage):
    """Changed files without valid receipts must not become durable completion markers."""
    await case.interrupt(stage)
    async with case.open() as app:

        async def reply(index, text):
            assert "do not append duplicate evidence" in text
            await case.write(app, index, f"Partial memory, attempt {len(case.calls)}")
            if bad_receipt == "exception":
                raise RuntimeError("incomplete integration")
            return {"result": "{}"}

        response = await case.run(app, reply=reply)
        assert not response.success
        assert len(case.calls) == (4 if stage == "after_extract" else 2)
        assert SOURCE not in response.metadata["dream"]["checkpoint_paths"]
        assert case.saved.completed == ([] if stage == "after_extract" else [0])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "rewrite", "later_interrupted_write"])
async def test_recheck_only_affected_digest_units(case, change):
    """Rechecking an earlier target also preserves evidence of a later interrupted write."""
    interrupted = change == "later_interrupted_write"
    await case.interrupt("after_write" if interrupted else "before_finish")
    first, second = (case.workspace / path for path in TARGETS)
    second_version = second.stat().st_mtime_ns
    if change == "delete":
        first.unlink()
    else:
        first.write_text("Overwritten memory")
    async with case.open() as app:

        async def reply(index, text):
            assert "do not append duplicate evidence" in text
            if interrupted:
                assert TARGETS[1] in text
            if index == 0:
                existed = first.exists()
                await case.write(app, index, UNITS[index]["summary"])
                return case.receipt(index, "CORRECT" if existed else "CREATE")
            assert interrupted
            return case.receipt(index, "CORROBORATE")

        response = await case.run(app, reply=reply)
        assert response.success, response.answer
        assert case.calls == (["first", "second"] if interrupted else ["first"])
        assert first.read_text().strip() == UNITS[0]["summary"]
        assert second.stat().st_mtime_ns == second_version


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["replacement", "catalog"])
async def test_failed_persistence_keeps_recoverable_checkpoint(case, failure):
    """Neither atomic replacement failure nor repeated catalog failure can discard progress."""
    async with case.open() as app:
        catalog = app.context.components["file_catalog"]["dream"]
        original_write, writes = CheckpointStore._write, 0

        def replace(store):
            nonlocal writes
            writes += 1
            with (
                (
                    patch("reme.steps.evolve.dream.checkpoint.os.replace", side_effect=OSError("disk full"))
                    if writes == 3
                    else nullcontext()
                ),
            ):
                return original_write(store)

        with (
            (
                patch.object(CheckpointStore, "_write", replace)
                if failure == "replacement"
                else patch.object(catalog, "dump", side_effect=OSError("disk full"))
            ),
        ):
            response = await case.run(app)
            assert not response.success and "disk full" in response.answer
            assert case.checkpoints
            if failure == "replacement":
                assert case.saved.active == 0 and case.saved.completed == []
                assert await catalog.get_nodes([SOURCE]) == []
                assert len(list(case.checkpoints[0].parent.iterdir())) == 1
            else:
                case.calls.clear()
                assert not (await case.run(app)).success
                assert not case.calls and case.checkpoints
        case.calls.clear()
        response = await case.run(app)
        assert response.success, response.answer
        assert "extract" not in case.calls and not case.checkpoints


@pytest.mark.asyncio
async def test_checkpoint_cannot_escape_workspace(case):
    """A metadata symlink must not allow checkpoint writes outside the workspace."""
    outside = case.workspace / "outside"
    outside.mkdir()
    nested = DreamCase(case.workspace / "nested")
    (nested.workspace / "metadata").symlink_to(outside, target_is_directory=True)
    async with nested.open() as app:
        response = await nested.run(app)
        assert not response.success and "checkpoint escapes workspace" in response.answer
        assert not nested.calls and not (outside / "dream").exists()


@pytest.mark.asyncio
async def test_edit_between_scan_and_extraction_keeps_source_eligible(case):
    """Do not attach a newer extraction to the scan's older source watermark."""

    def edit_then_hash(workspace, existing):
        previous = case.source.stat().st_mtime
        case.source.write_text("Modified after scanning")
        os.utime(case.source, (previous + 2, previous + 2))
        return source_versions(workspace, existing)

    async with case.open() as app:
        with patch("reme.steps.evolve.dream.extract.source_versions", edit_then_hash):
            response = await case.run(app)
        assert not response.success and "changed after scanning" in response.answer
        assert not case.calls
        assert await app.context.components["file_catalog"]["dream"].get_nodes([SOURCE]) == []
        assert (await case.run(app)).success
        assert case.calls == ["extract", "first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", ["integration", "finish"])
async def test_overlapping_jobs_do_not_replay_or_regress_progress(case, overlap):
    """Queued integrations refresh progress; stale finalizers cannot overwrite newer watermarks."""
    entered, release = asyncio.Event(), asyncio.Event()
    boundary = DreamIntegrateStep if overlap == "integration" else DreamFinishStep
    method = "_execute_locked" if overlap == "integration" else "execute"
    original, invocations = getattr(boundary, method), 0

    async def delay(step, *args):
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            entered.set()
            await release.wait()
        return await original(step, *args)

    async with case.open() as app:
        tasks = []
        try:
            with patch.object(boundary, method, delay):
                tasks.append(asyncio.create_task(case.run(app)))
                await asyncio.wait_for(entered.wait(), timeout=5)
                if overlap == "finish":
                    previous = case.source.stat().st_mtime
                    case.source.write_text("New source version")
                    os.utime(case.source, (previous + 2, previous + 2))
                    assert (await case.run(app)).success
                else:
                    restored = asyncio.Event()
                    restore = CheckpointStore.restore

                    def signal(store, sources):
                        result = restore(store, sources)
                        restored.set()
                        return result

                    with patch.object(CheckpointStore, "restore", signal):
                        tasks.append(asyncio.create_task(case.run(app)))
                        await asyncio.wait_for(restored.wait(), timeout=5)
                release.set()
                responses = await asyncio.gather(*tasks)
            if overlap == "finish":
                assert not responses[0].success and "another invocation" in responses[0].answer
                nodes = await app.context.components["file_catalog"]["dream"].get_nodes([SOURCE])
                assert nodes[0].st_mtime == case.source.stat().st_mtime
                case.calls.clear()
                assert (await case.run(app)).success and not case.calls
            else:
                assert any(r.success for r in responses)
                assert all(r.success or "another invocation" in r.answer for r in responses)
                assert case.calls == ["extract", "first", "second"]
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
