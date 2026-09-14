"""Unit tests for Claude Code auto-memory session persistence."""

# pylint: disable=protected-access

from types import SimpleNamespace
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.file_store import LocalFileStore
from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum
from reme.steps.evolve.auto_memory_cc import AutoMemoryCCStep


@pytest.mark.asyncio
async def test_reme_cc_store_preserves_existing_session_layout(tmp_path):
    """Existing transcript UUIDs remain visible at session/claude_code/<session_id>.jsonl."""
    step = AutoMemoryCCStep()
    step.file_store = SimpleNamespace(workspace_path=tmp_path)
    session_id = "session-1"
    session_path = tmp_path / "session" / "claude_code" / f"{session_id}.jsonl"
    session_path.parent.mkdir(parents=True)
    session_path.write_text('{"uuid":"existing"}\n', encoding="utf-8")

    increment = await step._save_cc_session(
        session_id,
        [{"uuid": "existing"}, {"uuid": "new"}],
    )

    assert increment == [{"uuid": "new"}]
    assert step._session_link(session_id) == f"[[session/claude_code/{session_id}.jsonl]]"
    assert not (tmp_path / "session" / "claude_code" / "claude_code").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("tag_fails", [False, True])
async def test_cc_job_tags_created_updated_and_renamed_notes_only(tmp_path, monkeypatch, tag_fails):
    """The configured CC job inherits changes, skips no-ops, and preserves its main result."""
    monkeypatch.chdir(tmp_path)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    store = LocalFileStore(name="default", embedding_store="", tag_index="default")
    app_context.components[ComponentEnum.FILE_STORE] = {"default": store}
    configs = resolve_app_config(config="default", log_config=False)["jobs"]
    session_id, day = "session-1", "2026-09-11"
    source = f"[[session/claude_code/{session_id}.jsonl]]"
    entries = []
    monkeypatch.setattr(AutoMemoryCCStep, "_load_cc_session", AsyncMock(return_value=entries))
    mode = "create"
    tag_paths = []
    memory_calls = []

    async def reply(_message, **kwargs):
        if "list_tags" in kwargs["job_tools"]:
            injected = kwargs["injected_job_kwargs"]
            path = injected["_allowed_paths"][0]
            tag_paths.append(path)
            if tag_fails:
                raise RuntimeError("tagging unavailable")
            result = await app_context.jobs["frontmatter_update"](
                path=path,
                metadata={"memory_tags": ["ReMe"]},
                **injected,
            )
            assert result.success
            return {"result": "Tagged ReMe"}
        memory_calls.append(mode)
        if mode == "create":
            target = tmp_path / f"daily/{day}/note.md"
            target.parent.mkdir(parents=True)
            post = frontmatter.Post("ReMe memory", name="note", session_id=session_id, source_conversation=source)
        elif mode == "unchanged":
            return {"result": "No new facts"}
        else:
            target = tmp_path / kwargs["injected_job_kwargs"]["_allowed_paths"][0]
            post = frontmatter.loads(target.read_text(encoding="utf-8"))
            post.content += "\nMore ReMe evidence"
            if mode == "rename":
                post.metadata["name"] = "renamed"
        target.write_text(frontmatter.dumps(post), encoding="utf-8")
        return {"result": "Recorded memory"}

    agent = AsyncMock(spec=BaseAgentWrapper)
    agent.reply.side_effect = reply
    await store.start()
    try:
        for name in ("daily_list", "move", "frontmatter_update", "auto_memory_cc"):
            job = BaseJob(
                name=name,
                steps=configs[name]["steps"],
                app_context=app_context,
                agent_wrapper=agent,
                date=day,
            )
            app_context.jobs[name] = job
            await job.start()
        job = app_context.jobs["auto_memory_cc"]
        for index, mode in enumerate(("create", "unchanged", "update", "rename")):
            entries.append(
                {"uuid": str(index), "type": "user", "message": {"role": "user", "content": f"ReMe fact {index}"}},
            )
            before_tags = len(tag_paths)
            response = await job(session_id=session_id)
            assert response.success is True
            assert response.metadata["source_conversation"] == source
            if mode == "unchanged":
                assert response.answer == "No new facts"
                assert response.metadata["modified"] is False
                assert response.metadata["auto_tag"]["processed"] == 0
                assert len(tag_paths) == before_tags
            else:
                path = f"daily/{day}/{'renamed' if mode == 'rename' else 'note'}.md"
                assert response.answer == "Recorded memory"
                assert response.metadata["path"] == path
                assert response.metadata["modified"] is True
                assert response.metadata["created"] is (mode == "create")
                tagging = response.metadata["auto_tag"]
                assert tagging["processed"] == 1
                assert tagging["failed"] == int(tag_fails)
                assert tagging["results"][0]["change"] == ("added" if mode == "create" else "modified")
                assert tag_paths[-1] == path
                post = frontmatter.loads((tmp_path / path).read_text(encoding="utf-8"))
                assert post.metadata.get("memory_tags") == (None if tag_fails else ["ReMe"])
                assert post.metadata["source_conversation"] == source

            # Repeating a Stop with the same UUIDs must run neither agent again.
            before_calls = (len(memory_calls), len(tag_paths))
            repeated = await job(session_id=session_id)
            assert repeated.success is True
            assert repeated.answer == "Skipped: no messages"
            assert repeated.metadata["auto_tag"]["processed"] == 0
            assert (len(memory_calls), len(tag_paths)) == before_calls
        assert not (tmp_path / f"daily/{day}/note.md").exists()
        assert not (tmp_path / "session/dialog").exists()
    finally:
        for job in reversed(list(app_context.jobs.values())):
            await job.close()
        await store.close()
