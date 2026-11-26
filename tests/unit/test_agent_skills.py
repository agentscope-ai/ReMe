"""Skill input, ownership and execution lifetime without model calls."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import frontmatter
import pytest
from pydantic import ValidationError

from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper, CcAgentWrapper
from reme.core.context import ApplicationContext
from reme.schema import SkillSpec, StreamChunk

# pylint: disable=missing-function-docstring,protected-access

GENERATED = {"name": "run-task", "description": "Run this task", "content": "# 工作流\nRead the source."}


class ProbeWrapper(BaseAgentWrapper):
    """Exercise the shared entry points while replacing the SDK boundary."""

    async def _create_session(self, **kwargs):
        kwargs["_skills"].publish(self.cwd / ".agents" / "skills")
        return "probe"

    async def _reply_stream(self, inputs, **kwargs):
        scope = kwargs["_skills"]
        scope.publish(self.cwd / ".agents" / "skills")
        assert Path(scope.directories[0], "SKILL.md").is_file()
        yield StreamChunk(chunk=inputs)
        if inputs == "fail":
            raise RuntimeError("model failed")
        if inputs == "wait":
            await asyncio.Event().wait()


def wrapper(tmp_path):
    return ProbeWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))


@pytest.mark.parametrize(
    "value",
    [
        "all",
        ["run-task"],
        None,
        [{"name": "../escape"}],
        [{"name": "x", "description": "missing body"}],
        [{"name": "x", "content": "body"}],
        [{"name": "x", "description": "d", "content": "  "}],
    ],
)
@pytest.mark.asyncio
async def test_invalid_definitions_fail_before_creating_files(tmp_path, value):
    with pytest.raises(ValidationError):
        async with wrapper(tmp_path).skill_scope(skills=value):
            pytest.fail("Invalid skills were accepted")
    assert not (tmp_path / "metadata").exists()


@pytest.mark.asyncio
async def test_scope_spans_session_creation_and_follow_up_calls(tmp_path):
    agent = wrapper(tmp_path)
    async with agent.skill_scope(skills=[SkillSpec(**GENERATED)]) as options:
        scope = options["_skills"]
        source = Path(scope.directories[0])
        manifest = source / "SKILL.md"
        note = frontmatter.loads(manifest.read_text())
        assert note["name"] == GENERATED["name"]
        assert note["description"] == GENERATED["description"]
        assert note.content == GENERATED["content"]
        original = manifest.stat().st_ino
        session_id = await agent.create_session(**options)
        for query in ("execute", "repair"):
            stream = agent.reply_stream(query, resume=session_id, **options)
            assert [chunk.chunk async for chunk in stream] == [query]
            assert manifest.stat().st_ino == original
    assert not source.exists()
    assert not (tmp_path / ".agents/skills/run-task").is_symlink()
    with pytest.raises(ValueError, match="current run"):
        await agent.create_session(**options)


@pytest.mark.asyncio
async def test_failed_stream_and_explicit_stream_close_clean_generated_files(tmp_path):
    agent = wrapper(tmp_path)
    with pytest.raises(RuntimeError, match="model failed"):
        _ = [chunk async for chunk in agent.reply_stream("fail", skills=[GENERATED])]
    assert not (tmp_path / "metadata/agent_skills/run-task").exists()
    stream = agent.reply_stream("wait", skills=[GENERATED])
    await anext(stream)
    assert (tmp_path / ".agents/skills/run-task/SKILL.md").is_file()
    await stream.aclose()
    assert not (tmp_path / ".agents/skills/run-task").is_symlink()
    assert not (tmp_path / "metadata/agent_skills/run-task").exists()


@pytest.mark.asyncio
async def test_cancelled_execution_closes_stream_before_skill_cleanup(tmp_path):
    agent = wrapper(tmp_path)
    ready = asyncio.Event()

    async def execute():
        async for _ in agent.reply_stream("wait", skills=[GENERATED]):
            ready.set()

    task = asyncio.create_task(execute())
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (tmp_path / "metadata/agent_skills/run-task").exists()
    assert not (tmp_path / ".agents/skills/run-task").is_symlink()


@pytest.mark.asyncio
async def test_concurrent_name_conflict_does_not_delete_the_first_run(tmp_path):
    agent = wrapper(tmp_path)
    async with agent.skill_scope(skills=[GENERATED]) as first:
        source = Path(first["_skills"].directories[0])
        with pytest.raises(FileExistsError):
            async with wrapper(tmp_path).skill_scope(skills=[GENERATED]):
                pytest.fail("Concurrent generated skill reused an existing name")
        assert (source / "SKILL.md").is_file()
    assert not source.exists()


@pytest.mark.asyncio
async def test_partial_preparation_and_publication_roll_back_only_owned_files(tmp_path):
    agent = wrapper(tmp_path)
    with pytest.raises(FileNotFoundError):
        async with agent.skill_scope(skills=[GENERATED, {"name": "missing"}]):
            pytest.fail("Missing file reference was accepted")
    assert not (tmp_path / "metadata/agent_skills/run-task").exists()
    target = tmp_path / ".agents/skills/run-task"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("User skill")
    with pytest.raises(FileExistsError):
        await agent.create_session(skills=[GENERATED])
    assert (target / "SKILL.md").read_text() == "User skill"
    assert not (tmp_path / "metadata/agent_skills/run-task").exists()


@pytest.mark.asyncio
async def test_existing_sources_and_duplicate_names_are_rejected_for_generated_skills(tmp_path):
    agent = wrapper(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        async with agent.skill_scope(skills=[GENERATED, GENERATED]):
            pytest.fail("Duplicate skill names were accepted")
    source = tmp_path / "skills/run-task"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("Original")
    with pytest.raises(FileExistsError):
        async with agent.skill_scope(skills=[GENERATED]):
            pytest.fail("Generated skill replaced a source")
    assert (source / "SKILL.md").read_text() == "Original"


@pytest.mark.asyncio
async def test_cleanup_preserves_a_replaced_discovery_link(tmp_path):
    agent = wrapper(tmp_path)
    target = tmp_path / ".agents/skills/run-task"
    with pytest.raises(ExceptionGroup, match="clean generated"):
        async with agent.skill_scope(skills=[GENERATED]) as options:
            await agent.create_session(**options)
            target.unlink()
            target.mkdir()
            (target / "keep").write_text("User replacement")
    assert (target / "keep").read_text() == "User replacement"
    assert not (tmp_path / "metadata/agent_skills/run-task").exists()


@pytest.mark.asyncio
async def test_agentscope_toolkit_loads_the_generated_file(tmp_path, monkeypatch):
    agent_wrapper = AsAgentWrapper(as_llm="", app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    agent_wrapper.as_llm = SimpleNamespace(model=object())
    monkeypatch.setattr(agent_wrapper, "_create_agent", SimpleNamespace)
    async with agent_wrapper.skill_scope(skills=[GENERATED]) as options:
        agent, _ = await agent_wrapper._build_agent("execute", **options)
        loaded = await agent.toolkit._get_available_skills()
        assert loaded[GENERATED["name"]].markdown == GENERATED["content"]
        assert loaded[GENERATED["name"]].description == GENERATED["description"]
    assert not Path(loaded[GENERATED["name"]].dir).exists()


@pytest.mark.parametrize("setting_source", ["user", "project"])
@pytest.mark.asyncio
async def test_claude_native_options_read_generated_skill_in_the_enabled_scope(tmp_path, monkeypatch, setting_source):
    from claude_agent_sdk import ResultMessage

    agent = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    seen = []

    async def messages(*, prompt, options):
        del prompt
        assert options.skills == [GENERATED["name"]]
        root = (
            Path(options.env["CLAUDE_CONFIG_DIR"]) / "skills"
            if setting_source == "user"
            else Path(options.cwd) / ".claude/skills"
        )
        manifest = root / GENERATED["name"] / "SKILL.md"
        assert frontmatter.loads(manifest.read_text()).content == GENERATED["content"]
        seen.append(manifest)
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=options.session_id,
            result="Done",
        )

    monkeypatch.setattr(agent, "_stream_messages", messages)
    assert (await agent.reply("execute", skills=[GENERATED], setting_sources=[setting_source]))["status"] == "completed"
    assert len(seen) == 1 and not seen[0].exists()
    assert not seen[0].parent.is_symlink()
    assert not (tmp_path / "metadata/agent_skills/run-task").exists()
