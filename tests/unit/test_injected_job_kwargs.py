"""Tests for the server-owned ``injected_job_kwargs`` wrapper mechanism.

Each agent wrapper merges these kwargs into every job tool call after
receiving the model's arguments, rejects model-supplied conflicts, and hides
the injected keys from the tool schema exposed to the model.
"""

# pylint: disable=protected-access,missing-function-docstring


import pytest

from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper, CcAgentWrapper
from reme.components.file_store import LocalFileStore
from reme.schema import Response
from reme.steps.evolve.auto_memory import AutoMemoryStep
from reme.steps.evolve.auto_memory_cc import AutoMemoryCCStep


class _Job:
    """Minimal job double recording every call."""

    def __init__(self, name="write"):
        self.name = name
        self.description = "Write a note"
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["path", "content", "date"],
        }
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return Response(answer="done")


# -- base helpers ---------------------------------------------------------------


def test_merge_rejects_model_supplied_conflicts():
    injected = {"_allowed_paths": ["daily/a.md"], "date": "2025-06-01"}
    merged = BaseAgentWrapper._merge_injected_job_kwargs({"path": "daily/a.md"}, injected)
    assert merged == {"path": "daily/a.md", "_allowed_paths": ["daily/a.md"], "date": "2025-06-01"}

    with pytest.raises(ValueError, match="cannot be provided by the model: _allowed_paths, date"):
        BaseAgentWrapper._merge_injected_job_kwargs(
            {"_allowed_paths": ["anywhere"], "date": "1999-01-01"},
            injected,
        )


def test_strip_injected_parameters_hides_keys_from_schema():
    job = _Job()
    stripped = BaseAgentWrapper._strip_injected_parameters(job.parameters, {"date": "2025-06-01"})
    assert "date" not in stripped["properties"]
    assert stripped["required"] == ["path", "content"]
    # Underscore keys never appear in schemas; stripping is a no-op then.
    untouched = BaseAgentWrapper._strip_injected_parameters(job.parameters, {"_allowed_paths": ["x"]})
    assert untouched["properties"].keys() == job.parameters["properties"].keys()
    # The original job schema is never mutated.
    assert "date" in job.parameters["properties"]


def test_search_injection_exposes_only_query():
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "min_score": {"type": "number"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
        },
        "required": ["query"],
    }
    injected = {
        "limit": 20,
        "min_score": 0.0,
        "start_date": None,
        "end_date": "2026-07-20",
    }

    stripped = BaseAgentWrapper._strip_injected_parameters(parameters, injected)

    assert stripped["properties"] == {"query": {"type": "string"}}
    assert stripped["required"] == ["query"]


# -- AgentScope wrapper -----------------------------------------------------------


@pytest.mark.asyncio
async def test_as_tool_injects_kwargs_and_rejects_conflicts():
    job = _Job()
    tool = AsAgentWrapper._make_tool(job, "ctx-1", {"_allowed_paths": ["daily/a.md"], "date": "2025-06-01"})

    assert "date" not in tool.input_schema["properties"]
    assert tool.input_schema["required"] == ["path", "content"]

    await tool.call(path="daily/a.md", content="hi")
    assert job.calls == [
        {
            "path": "daily/a.md",
            "content": "hi",
            "_allowed_paths": ["daily/a.md"],
            "date": "2025-06-01",
            "tool_context_id": "ctx-1",
        },
    ]

    with pytest.raises(ValueError, match="cannot be provided by the model"):
        await tool.call(path="daily/a.md", content="hi", _allowed_paths=["everything"])
    assert len(job.calls) == 1


@pytest.mark.asyncio
async def test_as_tool_without_injection_keeps_original_schema():
    job = _Job()
    tool = AsAgentWrapper._make_tool(job)
    assert tool.input_schema["required"] == ["path", "content", "date"]

    await tool.call(path="a.md", content="x", date="2025-06-01")
    assert job.calls == [{"path": "a.md", "content": "x", "date": "2025-06-01"}]


# -- Claude Code wrapper ----------------------------------------------------------


@pytest.mark.asyncio
async def test_cc_tool_injects_kwargs_and_rejects_conflicts():
    job = _Job()
    tool = CcAgentWrapper._make_tool(job, "ctx-1", {"_allowed_paths": ["daily/a.md"]})

    assert "_allowed_paths" not in tool.input_schema["properties"]

    result = await tool.handler({"path": "daily/a.md", "content": "hi", "date": "2025-06-01"})
    assert result["is_error"] is False
    assert job.calls == [
        {
            "path": "daily/a.md",
            "content": "hi",
            "date": "2025-06-01",
            "_allowed_paths": ["daily/a.md"],
            "tool_context_id": "ctx-1",
        },
    ]

    with pytest.raises(ValueError, match="cannot be provided by the model"):
        await tool.handler({"path": "daily/a.md", "content": "hi", "_allowed_paths": ["everything"]})
    assert len(job.calls) == 1


# -- AutoMemoryStep ---------------------------------------------------------------


class _RecordingWrapper(BaseAgentWrapper):
    """Agent wrapper double capturing reply() kwargs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[dict] = []

    async def reply(self, _inputs, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"session_id": "s-1", "last_message": {}, "result": "ok"}


async def _auto_memory_step(wrapper) -> AutoMemoryStep:
    store = LocalFileStore(name="t_inject", embedding_store="")  # workspace rooted at cwd
    await store.start()
    return AutoMemoryStep(name="auto_memory", agent_wrapper=wrapper, file_store=store)


@pytest.mark.asyncio
async def test_auto_memory_create_uses_model_supplied_date_and_daily_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper = _RecordingWrapper(name="fake")
    step = await _auto_memory_step(wrapper)

    async def no_note(_day, _session_id):
        return None

    monkeypatch.setattr(step, "_list_session_note", no_note)

    await step(
        session_id="sess-1",
        messages=[{"name": "user", "role": "user", "content": "hi", "created_at": "2025-06-01T10:00:00"}],
    )

    assert step.context.response.success is True
    assert wrapper.calls[0]["job_tools"] == ["daily_write"]
    assert "injected_job_kwargs" not in wrapper.calls[0]


@pytest.mark.asyncio
async def test_auto_memory_update_scopes_tools_to_exact_note_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper = _RecordingWrapper(name="fake")
    step = await _auto_memory_step(wrapper)
    session_id = "sess-1"
    note_path = "daily/2025-06-01/existing-note.md"
    note_file = tmp_path / note_path
    note_file.parent.mkdir(parents=True)
    note_file.write_text(
        f"---\nsession_id: {session_id}\nsource_conversation: '{step._session_link(session_id)}'\n---\nbody\n",
        encoding="utf-8",
    )

    async def existing_note(_day, _session_id):
        return {"path": note_path, "session_id": session_id}

    async def no_index(_store, _day, _daily_dir):
        return {}

    monkeypatch.setattr(step, "_list_session_note", existing_note)
    monkeypatch.setattr("reme.steps.evolve.auto_memory.refresh_day_index", no_index)

    await step(
        session_id=session_id,
        messages=[{"name": "user", "role": "user", "content": "hi again", "created_at": "2025-06-01T11:00:00"}],
    )

    assert step.context.response.success is True
    assert wrapper.calls[0]["job_tools"] == ["read", "edit", "frontmatter_update", "write"]
    assert wrapper.calls[0]["injected_job_kwargs"] == {"_allowed_paths": [note_path]}


def test_auto_memory_keeps_original_tool_names():
    """Core auto-memory uses the original file tool names."""
    step = AutoMemoryStep(name="auto_memory")
    assert step.create_tools == ["daily_write"]
    assert step.update_tools == ["read", "edit", "frontmatter_update", "write"]


def test_auto_memory_tags_are_disabled_by_default_and_enabled_through_kwargs():
    assert AutoMemoryStep()._tags_enabled() is False
    assert AutoMemoryCCStep()._tags_enabled() is False
    assert AutoMemoryStep(enable_tags=True)._tags_enabled() is True


def test_auto_memory_tags_prompt_follows_step_kwarg():
    step = AutoMemoryStep()
    prompt_kwargs = {
        "today": "2026-09-01",
        "note": "(none)",
        "session_id": "s1",
        "history": "user: remember GPT-5",
    }

    disabled_system = step.prompt_format("system_prompt", enable_tags=step._tags_enabled())
    disabled_create = step.prompt_format("user_message_create", enable_tags=step._tags_enabled(), **prompt_kwargs)
    disabled_update = step.prompt_format(
        "user_message_update",
        enable_tags=step._tags_enabled(),
        note_path="daily/2026-09-01/memory.md",
        **prompt_kwargs,
    )
    assert "`tags`" not in disabled_system
    assert 'metadata={"tags"' not in disabled_create
    assert '"tags"' not in disabled_update

    enabled_step = AutoMemoryStep(enable_tags=True)
    enabled_system = enabled_step.prompt_format("system_prompt", enable_tags=enabled_step._tags_enabled())
    enabled_create = enabled_step.prompt_format(
        "user_message_create",
        enable_tags=enabled_step._tags_enabled(),
        **prompt_kwargs,
    )
    enabled_update = enabled_step.prompt_format(
        "user_message_update",
        enable_tags=enabled_step._tags_enabled(),
        note_path="daily/2026-09-01/memory.md",
        **prompt_kwargs,
    )
    assert "`tags`" in enabled_system
    assert 'metadata={"tags"' in enabled_create
    assert '"tags"' in enabled_update


def test_auto_memory_create_prompts_match_upstream_date_arguments():
    """Auto-memory prompts keep the upstream model-supplied date argument."""
    from pathlib import Path

    prompt_file = Path("reme/steps/evolve/auto_memory.yaml")
    evolve_prompt = prompt_file.read_text(encoding="utf-8")
    assert "date={today}" in evolve_prompt or "`date`: {today}" in evolve_prompt or "`date`：{today}" in evolve_prompt
    assert '"tags": [<tag>, ...]' in evolve_prompt
    assert '"tags": []' in evolve_prompt or "using `[]`" in evolve_prompt


def test_configs_define_original_jobs_without_daily_variants():
    from reme.config import resolve_app_config

    for config_name in ("default",):
        config = resolve_app_config(config=config_name, log_config=False)
        jobs = config["jobs"]
        for name in ("read", "edit", "write", "frontmatter_update", "daily_write"):
            assert name in jobs, f"{config_name} missing job {name}"
        for name in ("read_daily", "edit_daily", "write_daily"):
            assert name not in jobs, f"{config_name} unexpectedly defines {name}"

    default = resolve_app_config(config="default", log_config=False)
    assert default["jobs"]["auto_memory"]["steps"][0]["enable_tags"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
