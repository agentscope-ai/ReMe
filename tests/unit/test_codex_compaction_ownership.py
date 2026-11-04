"""Compaction claims a Codex thread before any SDK side effect."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
import json
from types import SimpleNamespace

import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper


def _owned_on_disk(tmp_path, alias) -> set[str]:
    path = tmp_path / alias / "mem_session" / "codex" / "owned_threads.json"
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8"))["thread_ids"])


@pytest.mark.asyncio
async def test_compact_arbitrates_before_sdk_side_effects(tmp_path, monkeypatch):
    """Compaction claims ownership before resume/compact, including under concurrency."""
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    alpha = CodexAgentWrapper(name="codex-alpha", app_context=context, release_threads=False)
    beta = CodexAgentWrapper(name="codex-beta", app_context=context, release_threads=False)
    sdk_ops: list[str] = []
    entered_sdk = asyncio.Event()
    release_sdk = asyncio.Event()

    class PausingThread:
        """Resume pauses inside the SDK so a competing resume can try to claim."""

        id = "shared-thread"

        async def compact(self):
            sdk_ops.append("compact")

    async def alpha_resume(_thread_id):
        sdk_ops.append("alpha-resume")
        entered_sdk.set()
        await release_sdk.wait()
        return PausingThread()

    async def beta_resume(thread_id, **_kwargs):
        sdk_ops.append("beta-resume")
        return SimpleNamespace(id=thread_id)

    async def alpha_codex():
        return SimpleNamespace(thread_resume=alpha_resume)

    monkeypatch.setattr(alpha, "_get_codex", alpha_codex)
    compact_task = asyncio.create_task(alpha.compact_session("shared-thread", workspace="alpha"))
    await asyncio.wait_for(entered_sdk.wait(), timeout=2)

    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await beta._open_thread(
            SimpleNamespace(thread_resume=beta_resume),
            {"session_id": "shared-thread", "workspace": "beta"},
        )

    release_sdk.set()
    await asyncio.wait_for(compact_task, timeout=2)
    assert sdk_ops == ["alpha-resume", "compact"]
    assert context.workspaces["alpha"].codex_thread_state.owned_thread_ids == {"shared-thread"}
    assert "shared-thread" not in (context.workspaces["beta"].codex_thread_state.owned_thread_ids or set())
    assert _owned_on_disk(tmp_path, "alpha") == {"shared-thread"}
    assert "shared-thread" not in _owned_on_disk(tmp_path, "beta")

    # An alias that was never configured fails before either SDK method.
    unknown_ops: list[str] = []

    async def unknown_resume(_thread_id):
        unknown_ops.append("resume")
        return PausingThread()

    async def unknown_codex():
        return SimpleNamespace(thread_resume=unknown_resume)

    monkeypatch.setattr(alpha, "_get_codex", unknown_codex)
    with pytest.raises(ValueError, match="Unknown workspace 'typo'"):
        await alpha.compact_session("unrecorded-thread", workspace="typo")
    assert not unknown_ops
    assert "unrecorded-thread" not in (context.workspaces["alpha"].codex_thread_state.owned_thread_ids or set())

    # Once beta owns a thread, alpha's compaction does not reach the SDK.
    await beta._open_thread(
        SimpleNamespace(thread_resume=beta_resume),
        {"session_id": "beta-thread", "workspace": "beta"},
    )
    loser_ops: list[str] = []

    async def loser_resume(_thread_id):
        loser_ops.append("resume")
        return PausingThread()

    async def loser_codex():
        return SimpleNamespace(thread_resume=loser_resume)

    monkeypatch.setattr(alpha, "_get_codex", loser_codex)
    with pytest.raises(ValueError, match="belongs to workspace 'beta'"):
        await alpha.compact_session("beta-thread", workspace="alpha")
    assert not loser_ops
    assert context.workspaces["beta"].codex_thread_state.owned_thread_ids == {"beta-thread"}


@pytest.mark.asyncio
async def test_compact_drops_new_claim_when_sdk_rejects(tmp_path, monkeypatch):
    """A failed resume or compact releases a claim this call just made."""
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    alpha = CodexAgentWrapper(name="codex-alpha", app_context=context)
    beta = CodexAgentWrapper(name="codex-beta", app_context=context, release_threads=False)
    released: list[str] = []

    class RecordingClient:
        """Capture thread ids passed to the SDK unsubscribe request."""

        async def request(self, method, params, *, response_model):
            released.append(params["threadId"])
            assert method == "thread/unsubscribe"
            assert response_model is not None
            return SimpleNamespace()

    class RejectingThread:
        """Resume succeeds and compaction is rejected by the SDK."""

        def __init__(self, thread_id):
            self.id = thread_id

        async def compact(self):
            raise RuntimeError("sdk rejected")

    async def resume(thread_id):
        if thread_id == "resume-fails":
            raise RuntimeError("resume rejected")
        return RejectingThread(thread_id)

    async def get_codex():
        return SimpleNamespace(_client=RecordingClient(), thread_resume=resume)

    monkeypatch.setattr(alpha, "_get_codex", get_codex)

    with pytest.raises(RuntimeError, match="resume rejected"):
        await alpha.compact_session("resume-fails", workspace="alpha")
    with pytest.raises(RuntimeError, match="sdk rejected"):
        await alpha.compact_session("new-thread", workspace="alpha")
    await alpha._record_thread_ownership("alpha", "kept-thread")
    with pytest.raises(RuntimeError, match="sdk rejected"):
        await alpha.compact_session("kept-thread", workspace="alpha")

    owned = context.workspaces["alpha"].codex_thread_state.owned_thread_ids or set()
    assert "resume-fails" not in owned
    assert "new-thread" not in owned
    assert owned == {"kept-thread"}
    assert _owned_on_disk(tmp_path, "alpha") == {"kept-thread"}
    assert released == ["new-thread", "kept-thread"]

    await beta._record_thread_ownership("beta", "new-thread")
    assert context.workspaces["beta"].codex_thread_state.owned_thread_ids == {"new-thread"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("worker_operation", ["reply", "compact"])
async def test_failed_compactor_keeps_ownership_confirmed_by_another_wrapper(
    tmp_path,
    monkeypatch,
    cancel,
    worker_operation,
):
    """Failure/cancellation cannot revoke another wrapper's successful use."""
    config = {"workspace_dir": {"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")}}
    context = ApplicationContext(**config)
    compactor = CodexAgentWrapper(name="compactor", app_context=context, release_threads=False)
    worker = CodexAgentWrapper(name="worker", app_context=context, release_threads=False)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def failing_client():
        entered.set()
        await finish.wait()
        raise RuntimeError("client initialization failed")

    async def run(_inputs, **_kwargs):
        return SimpleNamespace(final_response="ok")

    async def compact():
        return None

    async def resume(thread_id, **_kwargs):
        return SimpleNamespace(id=thread_id, run=run, compact=compact)

    async def working_client():
        return SimpleNamespace(thread_resume=resume)

    monkeypatch.setattr(compactor, "_get_codex", failing_client)
    monkeypatch.setattr(worker, "_get_codex", working_client)
    monkeypatch.setattr(worker, "_serialize", lambda _result: {})
    task = asyncio.create_task(compactor.compact_session("T", workspace="alpha"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        if worker_operation == "reply":
            result = await worker.reply("hello", resume="T", workspace="alpha")
            assert result["result"] == "ok"
        else:
            await worker.compact_session("T", workspace="alpha")
        if cancel:
            task.cancel()
        else:
            finish.set()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await asyncio.wait_for(task, timeout=2)

        assert context.workspaces["alpha"].codex_thread_state.owned_thread_ids == {"T"}
        assert not context.workspaces["alpha"].codex_thread_state.provisional_claims
        assert _owned_on_disk(tmp_path, "alpha") == {"T"}
        for checked_context in (context, ApplicationContext(**config)):
            beta = CodexAgentWrapper(name="beta", app_context=checked_context, release_threads=False)
            monkeypatch.setattr(beta, "_get_codex", working_client)
            with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
                await beta.reply("hello", resume="T", workspace="beta")
        assert not _owned_on_disk(tmp_path, "beta")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await compactor.close()
        await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_compactor_retains_another_pending_reservation(tmp_path, monkeypatch, cancel):
    """The last failed reservation cleans up; earlier failures preserve active work."""
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    wrappers = [CodexAgentWrapper(name=name, app_context=context) for name in ("first", "second")]
    entered = [asyncio.Event(), asyncio.Event()]
    finish = [asyncio.Event(), asyncio.Event()]

    def client(index):
        async def get_client():
            entered[index].set()
            await finish[index].wait()
            raise RuntimeError("failed")

        return get_client

    tasks = []
    try:
        for index, wrapper in enumerate(wrappers):
            monkeypatch.setattr(wrapper, "_get_codex", client(index))
            tasks.append(asyncio.create_task(wrapper.compact_session("T", workspace="alpha")))
            await asyncio.wait_for(entered[index].wait(), timeout=2)
        if cancel:
            tasks[0].cancel()
        else:
            finish[0].set()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await asyncio.wait_for(tasks[0], timeout=2)
        assert _owned_on_disk(tmp_path, "alpha") == {"T"}
        beta = CodexAgentWrapper(app_context=context)
        with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
            await beta.compact_session("T", workspace="beta")
        finish[1].set()
        with pytest.raises(RuntimeError, match="failed"):
            await asyncio.wait_for(tasks[1], timeout=2)
        assert context.workspaces["alpha"].codex_thread_state.owned_thread_ids == set()
        assert not context.workspaces["alpha"].codex_thread_state.provisional_claims
        assert not _owned_on_disk(tmp_path, "alpha")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for wrapper in wrappers:
            await wrapper.close()
