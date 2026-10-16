"""Context Agent evidence, decision, lifecycle and host-delivery contracts."""

# pylint: disable=protected-access,redefined-outer-name,missing-function-docstring

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from reme.application import Application
from reme.components import ApplicationContext, R
from reme.components.as_classifier import BaseAsClassifier
from reme.components.context_agent import ContextAgent
from reme.components.file_chunker import MarkdownFileChunker
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum
from reme.schema import ContextAgentOptions, ContextEvaluationResult, ContextEvent, FileChunk, FileNode
from reme.steps.context import EvaluateContextStep
from reme.steps.context.candidates import discover_candidates
from reme.steps.context.evaluate_context import evaluate_context
from reme.steps.context.policy import answer_probabilities, fit_request


def event(revision=1, **snapshot):
    return ContextEvent(
        session_id="session",
        task_id="task",
        event_id=f"event-{revision}",
        revision=revision,
        event_type="tool_start",
        task_snapshot={"goal": "Diagnose Atlas production proxy failure", **snapshot},
    )


class Classifier(BaseAsClassifier):
    """Typed recording test boundary; no provider or network calls."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = None
        self.call_count = 0
        self.states = []
        self.probability = 0.95
        self.error = None

    async def __call__(self, state, questions):
        self.call_count += 1
        self.states.append(copy.deepcopy(state))
        self.started.set()
        if self.release:
            await self.release.wait()
        if self.error:
            raise self.error
        return SimpleNamespace(
            content={key: {"type": "binary_answer", "probability": self.probability} for key in questions},
            usage=None,
        )


def seed(workspace, text="Atlas production requires the proxy gateway, not direct egress.", path="memory/atlas.md"):
    target = workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return FileChunk(path=path, start_line=1, end_line=len(text.splitlines()), text=text).set_hash_id()


def store_for(chunks, workspace):
    return SimpleNamespace(
        vector_search=AsyncMock(return_value=[]),
        keyword_search=AsyncMock(return_value=chunks),
        get_nodes=AsyncMock(
            return_value=[FileNode(path=c.path, st_mtime=(workspace / c.path).stat().st_mtime) for c in chunks],
        ),
    )


@pytest.fixture
def environment(tmp_path):
    classifier = Classifier()
    chunk = seed(tmp_path)
    store = store_for([chunk], tmp_path)
    return tmp_path, classifier, chunk, store


def coordinator(workspace, classifier, store, **limits):
    agent = ContextAgent(
        app_context=ApplicationContext(workspace_dir=str(workspace)),
        options={"total_timeout_ms": 2000},
        limits={"enabled": True, "mode": "hint", "cooldown_seconds": 0, **limits},
    )
    agent.as_classifier, agent.file_store = classifier, store
    return agent


def test_evaluation_returns_real_sources_and_preserves_inputs(environment):
    async def run():
        workspace, classifier, chunk, store = environment
        snapshot = event()
        original = snapshot.model_dump()
        result = await evaluate_context(
            snapshot,
            ContextAgentOptions(),
            store=store,
            classifier=classifier,
            workspace=workspace,
        )
        assert result.action == "hint"
        assert result.hint.evidence[0].path == chunk.path
        assert result.hint.evidence[0].excerpt == chunk.text
        assert chunk.path in result.hint.text
        assert result.stats["classifier_calls"] == 1
        assert snapshot.model_dump() == original
        assert "tool_contexts" not in store.__dict__
        assert ContextEvaluationResult.model_validate_json(result.model_dump_json()).action == "hint"

    asyncio.run(run())


@pytest.mark.parametrize("restriction", [[], ["../outside"], ["other"], "~", 123])
def test_permissions_fail_closed_before_classifier(environment, restriction):
    async def run():
        workspace, classifier, _chunk, store = environment
        result = await evaluate_context(
            event(),
            ContextAgentOptions(),
            store=store,
            classifier=classifier,
            workspace=workspace,
            allowed_paths=restriction,
        )
        assert result.reason_code == "no_candidates"
        assert classifier.call_count == 0

    asyncio.run(run())


def test_escape_symlink_stale_index_and_oversized_sources_are_not_evidence(environment, tmp_path):
    async def run():
        workspace, classifier, chunk, store = environment
        outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
        outside.write_text("private outside source", encoding="utf-8")
        try:
            (workspace / "escape.md").symlink_to(outside)
            store.keyword_search.return_value = [
                chunk.model_copy(update={"text": "stale cached text"}),
                chunk.model_copy(update={"path": str(outside)}),
                chunk.model_copy(update={"path": "escape.md"}),
                chunk.model_copy(update={"path": "../outside.md"}),
            ]
            store.get_nodes.return_value[0].st_mtime = 0
            assert await discover_candidates(store, workspace, event(), ContextAgentOptions(), None) == []
            (workspace / chunk.path).write_text("x" * 2048)
            store.keyword_search.return_value = [chunk]
            options = ContextAgentOptions(max_file_bytes=1024)
            assert await discover_candidates(store, workspace, event(), options, None) == []
            assert classifier.call_count == 0
        finally:
            outside.unlink()

    asyncio.run(run())


def test_known_and_low_value_evidence_stays_silent(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        options = ContextAgentOptions()
        candidates = await discover_candidates(store, workspace, event(), options, None)
        result = await evaluate_context(
            event(seen_evidence=[candidates[0].candidate_id]),
            options,
            store=store,
            classifier=classifier,
            workspace=workspace,
        )
        assert result.action == "silent" and classifier.call_count == 0
        classifier.probability = 0.2
        result = await evaluate_context(event(), options, store=store, classifier=classifier, workspace=workspace)
        assert result.reason_code == "no_incremental_value"

    asyncio.run(run())


def test_standard_markdown_chunks_quote_original_lines(tmp_path):
    async def run():
        original = "# Atlas\n\n## Production\n\nUse **CONNECT** with the original hostname.\n\n- Preserve SNI.\n"
        chunk = seed(tmp_path, original)
        context = ApplicationContext(workspace_dir=str(tmp_path))
        node, chunks = await MarkdownFileChunker(app_context=context).chunk(tmp_path / chunk.path)
        store = store_for(chunks, tmp_path)
        store.get_nodes.return_value = [node]
        candidates = await discover_candidates(store, tmp_path, event(), ContextAgentOptions(), None)
        assert candidates
        for candidate in candidates:
            assert candidate.excerpt == "\n".join(
                original.splitlines()[candidate.start_line - 1 : candidate.end_line],
            )

    asyncio.run(run())


@pytest.mark.parametrize("probability", [None, True, -0.1, 1.1, float("nan"), float("inf"), "0.8"])
def test_invalid_classifier_answers_are_rejected(probability):
    response = SimpleNamespace(content={"q": {"type": "binary_answer", "probability": probability}})
    with pytest.raises(ValueError, match="Invalid"):
        answer_probabilities(response, {"q": {"type": "binary"}})


@pytest.mark.parametrize("content", [{}, {"q": {"type": "refusal_answer", "probability": 1}}])
def test_missing_or_refused_answers_are_rejected(content):
    with pytest.raises(ValueError):
        answer_probabilities(SimpleNamespace(content=content), {"q": {"type": "binary"}})


def test_request_budget_drops_whole_candidates_and_preserves_goal(environment):
    async def run():
        workspace, _classifier, _chunk, store = environment
        candidates = await discover_candidates(store, workspace, event(), ContextAgentOptions(), None)
        many = candidates * 8
        state, questions, fitted, estimate = fit_request(event(), many, ContextAgentOptions())
        assert 0 < len(fitted) < len(many)
        assert estimate <= 4000
        assert state["task"]["goal"] == event().task_snapshot.goal
        assert len(questions) == 1 + 2 * len(fitted)
        with pytest.raises(ValueError, match="budget"):
            fit_request(event(recent_context="x" * 6000), candidates, ContextAgentOptions())

    asyncio.run(run())


def test_source_changed_while_classifier_runs_is_not_delivered(environment):
    async def run():
        workspace, classifier, chunk, store = environment
        classifier.release = asyncio.Event()
        task = asyncio.create_task(
            evaluate_context(
                event(),
                ContextAgentOptions(total_timeout_ms=2000),
                store=store,
                classifier=classifier,
                workspace=workspace,
            ),
        )
        await classifier.started.wait()
        (workspace / chunk.path).write_text("changed source")
        classifier.release.set()
        result = await task
        assert result.reason_code == "source_changed"
        assert result.hint is None

    asyncio.run(run())


def test_timeout_and_cancellation_propagate(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        classifier.release = asyncio.Event()
        with pytest.raises(TimeoutError):
            await evaluate_context(
                event(),
                ContextAgentOptions(total_timeout_ms=20),
                store=store,
                classifier=classifier,
                workspace=workspace,
            )
        task = asyncio.create_task(
            evaluate_context(
                event(),
                ContextAgentOptions(total_timeout_ms=2000),
                store=store,
                classifier=classifier,
                workspace=workspace,
            ),
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_coordinator_is_nonblocking_coalesces_and_leases_before_ack(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        classifier.release = asyncio.Event()
        agent = coordinator(workspace, classifier, store)
        async with agent:
            assert await agent.submit(event())
            await classifier.started.wait()
            # No model wait at the tool-result/model-request boundary.
            assert await agent.take_ready("session", "task", 1) is None
            assert await agent.submit(event(2))
            assert await agent.submit(event(3))
            assert not await agent.submit(event(2))
            classifier.release.set()
            await agent.wait_idle("session")
            assert classifier.call_count == 2
            assert await agent.take_ready("session", "task", 1) is None
            hint = await agent.take_ready("session", "task", 3)
            assert hint and hint.based_on_revision == 3
            assert await agent.take_ready("session", "task", 3) is None
            assert not await agent.ack("other-session", hint.hint_id)
            assert await agent.ack("session", hint.hint_id)
            assert not await agent.ack("session", hint.hint_id)
            await agent.submit(event(4))
            await agent.wait_idle("session")
            assert (await agent.last_result("session")).reason_code == "no_candidates"
            assert classifier.call_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("changed", ["source", "permissions", "expired"])
def test_delivery_rechecks_current_source_scope_and_ttl(environment, changed):
    async def run():
        workspace, classifier, chunk, store = environment
        agent = coordinator(workspace, classifier, store)
        async with agent:
            await agent.submit(event(), allowed_paths=["memory"])
            await agent.wait_idle("session")
            if changed == "source":
                (workspace / chunk.path).write_text("new facts")
            if changed == "expired":
                agent._sessions["session"].result.hint.expires_at = 1
            scope = [] if changed == "permissions" else ["memory"]
            assert await agent.take_ready("session", "task", 1, allowed_paths=scope) is None

    asyncio.run(run())


def test_shadow_mode_and_disabled_mode_never_inject(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        agent = coordinator(workspace, classifier, store, mode="shadow")
        async with agent:
            await agent.submit(event())
            await agent.wait_idle("session")
            assert (await agent.last_result("session")).action == "hint"
            assert await agent.take_ready("session", "task", 1) is None
        disabled = coordinator(workspace, classifier, store, enabled=False)
        async with disabled:
            assert not await disabled.submit(event())

    asyncio.run(run())


def test_close_reclaims_inflight_and_restart_is_empty(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        classifier.release = asyncio.Event()
        agent = coordinator(workspace, classifier, store)
        await agent.start()
        await agent.submit(event())
        await classifier.started.wait()
        await agent.close()
        assert not agent._tasks and not agent._sessions
        assert not await agent.submit(event(2))
        await agent.start()
        assert await agent.last_result("session") is None
        await agent.close()

    asyncio.run(run())


def test_resource_and_evaluation_limits(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        agent = coordinator(workspace, classifier, store, max_sessions=1, max_evaluations=1)
        async with agent:
            await agent.submit(event())
            await agent.wait_idle("session")
            assert not await agent.submit(event().model_copy(update={"session_id": "other"}))
            await agent.submit(event(2))
            await agent.wait_idle("session")
            assert classifier.call_count == 1
            await agent.dispose_session("session")
            assert await agent.submit(event().model_copy(update={"session_id": "other"}))
            await agent.wait_idle("other")

    asyncio.run(run())


def test_global_concurrency_and_dispose_cancel_queued_work(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        classifier.release = asyncio.Event()
        agent = coordinator(workspace, classifier, store, max_concurrency=1)
        async with agent:
            await agent.submit(event())
            await classifier.started.wait()
            await agent.submit(event().model_copy(update={"session_id": "queued"}))
            await asyncio.sleep(0)
            assert classifier.call_count == 1
            await agent.dispose_session("queued")
            assert await agent.last_result("queued") is None
            classifier.release.set()
            await agent.wait_idle("session")
            assert classifier.call_count == 1

    asyncio.run(run())


def test_coordinator_timeout_is_silent_and_has_no_orphan_tasks(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        classifier.release = asyncio.Event()
        agent = coordinator(workspace, classifier, store)
        agent.options.total_timeout_ms = 20
        async with agent:
            await agent.submit(event())
            await agent.wait_idle("session")
            result = await agent.last_result("session")
            assert result.action == "silent" and result.reason_code == "timeout"
            assert await agent.take_ready("session", "task", 1) is None
            assert not agent._tasks

    asyncio.run(run())


def test_delivery_retry_and_hint_budget(environment):
    async def run():
        workspace, classifier, _chunk, store = environment
        agent = coordinator(workspace, classifier, store, max_hints=1)
        async with agent:
            await agent.submit(event())
            await agent.wait_idle("session")
            first = await agent.take_ready("session", "task", 1)
            agent._sessions["session"].lease_until = 0
            retry = await agent.take_ready("session", "task", 1)
            assert first.hint_id == retry.hint_id
            assert await agent.ack("session", retry.hint_id)
            await agent.submit(event(2))
            await agent.wait_idle("session")
            assert classifier.call_count == 1

    asyncio.run(run())


def test_job_registration_private_config_and_default_remains_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("CLASSIFIER_API_KEY", raising=False)
    config = resolve_app_config(config="context_agent", log_config=False)
    assert R.get(ComponentEnum.CONTEXT_AGENT, "jev") is ContextAgent
    assert R.get(ComponentEnum.STEP, "evaluate_context_step") is EvaluateContextStep
    assert not config["jobs"]["evaluate_context"]["enable_serve"]
    assert "context_agent" not in resolve_app_config(log_config=False)["components"]

    async def run():
        # Keep only local components and the private Job; no watchers or remote models.
        config["jobs"] = {"evaluate_context": config["jobs"]["evaluate_context"]}
        config.update(workspace_dir=str(tmp_path), enable_logo=False, log_to_console=False, log_to_file=False)
        async with Application(**config) as app:
            store = app.context.components["file_store"]["default"]
            chunk = seed(tmp_path)
            await store.upsert([(FileNode(path=chunk.path, st_mtime=(tmp_path / chunk.path).stat().st_mtime), [chunk])])
            fake = Classifier()
            app.context.components["as_classifier"]["context_agent"] = fake
            result = await app.run_job("evaluate_context", event=event().model_dump())
            assert result.success and result.answer["action"] == "hint"
            assert result.metadata["context_agent"]["action"] == "hint"
            bad = await app.run_job("evaluate_context", event={})
            assert not bad.success

    asyncio.run(run())


@pytest.mark.parametrize("options", [{"total_timeout_ms": 0}, {"candidate_threshold": float("nan")}, {"key": "no"}])
def test_options_reject_invalid_configuration(options):
    with pytest.raises(ValidationError):
        ContextAgentOptions.model_validate(options)
