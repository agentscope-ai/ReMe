"""Focused tests for the Meta-ReMe AgentScope agents and diagnostic tool."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest
from agentscope.permission import PermissionMode
from agentscope.state import AgentState

META_REME = Path(__file__).resolve().parents[2] / "meta-reme"
if str(META_REME) not in sys.path:
    sys.path.insert(0, str(META_REME))

utils = importlib.import_module("as.utils")
diagnostic_agent_module = importlib.import_module("as.agent.diagnostic_agent")
optimizer_agent_module = importlib.import_module("as.agent.optimizer_agent")
diagnostic_tool_module = importlib.import_module("as.tools.diagnostic_tool")
validation_tool_module = importlib.import_module("as.tools.validation_tool")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _workspace(tmp_path: Path) -> tuple[Path, str, Path]:
    workspace = tmp_path / "workspace"
    repository = workspace / "code/repo/reme"
    repository.mkdir(parents=True)
    (workspace / "logs").mkdir()
    (workspace / "dataset/cases").mkdir(parents=True)
    (workspace / ".meta-reme-workspace.json").write_text("{}\n", encoding="utf-8")
    (repository / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "init", "--initial-branch", "init")
    _git(repository, "add", "candidate.py")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    commit = _git(repository, "rev-parse", "HEAD")
    validation = workspace / f"evaluations/init/{commit}/initial"
    validation.mkdir(parents=True)
    (validation / "manifest.json").write_text(json.dumps({"branch": "init", "commit": commit}), encoding="utf-8")
    (validation / "summary.json").write_text(
        json.dumps({"status": "completed", "case_count": 1, "mean_query_score": 0.5}),
        encoding="utf-8",
    )
    return workspace, commit, validation


def test_prompt_files_are_loaded_by_same_stem() -> None:
    diagnostic = utils.load_agent_prompt(META_REME / "as/agent/diagnostic_agent.py")
    optimizer = utils.load_agent_prompt(META_REME / "as/agent/optimizer_agent.py")

    assert diagnostic.name == "reme_code_diagnostician"
    assert diagnostic.model == "qwen3.8-max"
    assert "{workspace}" in diagnostic.system_prompt
    assert optimizer.name == "reme_benchmark_optimizer"
    assert optimizer.model == "qwen3.8-max"
    assert "diagnostic_subagent_tool" in optimizer.system_prompt
    assert "complete behavior path" in optimizer.system_prompt
    assert "before making a prompt-only candidate" in optimizer.system_prompt
    assert "distinct evidence-backed mechanisms" in optimizer.system_prompt
    assert "does not need to run every" in optimizer.system_prompt
    assert "Preserve generalization beyond" in optimizer.system_prompt
    assert "checked out at that winning branch" in optimizer.system_prompt


def test_validation_catalog_and_git_tools_cover_versioned_results(tmp_path: Path) -> None:
    workspace, commit, validation = _workspace(tmp_path)

    catalog = utils.validation_catalog(workspace)
    tools = utils.create_git_inspection_tools(workspace)
    history = tools[0]._func(max_count=5)
    comparison = tools[1]._func(commit, commit)

    assert catalog["validation_count"] == 1
    assert catalog["validations"][0]["details_path"] == str(validation)
    assert catalog["validations"][0]["summary"]["mean_query_score"] == 0.5
    assert commit in history["history"]
    assert comparison["base_commit"] == commit
    assert comparison["target_commit"] == commit
    assert comparison["diff"] == ""


def test_memory_tools_inspect_session_history_and_compare_checkpoints(tmp_path: Path) -> None:
    workspace, _, validation = _workspace(tmp_path)
    repository = validation / "cases/case-1/memory_construction/reme_workspace"
    daily = repository / "daily"
    daily.mkdir(parents=True)
    _git(repository, "init", "--initial-branch", "main")
    (daily / "memory.md").write_text("first\n", encoding="utf-8")
    _git(repository, "add", "daily")
    _git(repository, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "session: one")
    base = _git(repository, "rev-parse", "HEAD")
    (daily / "memory.md").write_text("first\nsecond\n", encoding="utf-8")
    _git(repository, "add", "daily")
    _git(repository, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "session: two")
    target = _git(repository, "rev-parse", "HEAD")

    listing, inspection, comparison, snapshots = utils.create_memory_inspection_tools(workspace)
    histories = listing._func(str(validation))
    history = inspection._func(str(validation), "case-1", 10, "daily")
    diff = comparison._func(str(validation), "case-1", base, target, "daily")
    snapshot_diff = snapshots._func(str(validation), str(validation), "case-1", "daily")

    assert histories["histories"][0]["checkpoint_count"] == 2
    assert "session: two" in history["history"]
    assert diff["base_commit"] == base
    assert diff["target_commit"] == target
    assert "+second" in diff["diff"]
    assert snapshot_diff["diff"] == ""


def test_memory_tools_reject_paths_outside_evaluations(tmp_path: Path) -> None:
    workspace, _, _ = _workspace(tmp_path)
    listing = utils.create_memory_inspection_tools(workspace)[0]

    with pytest.raises(ValueError, match="outside this workspace"):
        listing._func(str(tmp_path))


@pytest.mark.asyncio
async def test_diagnostic_tool_injects_full_trajectory_and_validation_paths(tmp_path: Path, monkeypatch) -> None:
    workspace, _, validation = _workspace(tmp_path)
    observed = {}

    class FakeReply:
        """Minimal message compatible with ``message_text``."""

        def get_text_content(self):
            return "root cause report"

    async def fake_run(agent, *, request, trajectory_path, validation_paths):
        del agent
        observed.update(
            request=request,
            trajectory_path=trajectory_path,
            validation_paths=validation_paths,
        )
        return FakeReply()

    monkeypatch.setattr(diagnostic_tool_module, "create_diagnostic_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(diagnostic_tool_module, "run_diagnostic_agent", fake_run)
    tool = diagnostic_tool_module.create_diagnostic_subagent_tool(object(), workspace)
    state = AgentState()
    state.middle_context["marker"] = {"complete": True}

    chunk = await tool.call(
        instruction="compare regression",
        validation_paths=[str(validation)],
        trajectory="runtime event stream",
        _agent_state=state,
    )
    result = json.loads(chunk.content[0].text)
    saved = json.loads(Path(result["trajectory_path"]).read_text(encoding="utf-8"))

    assert tool.is_state_injected is True
    assert "_agent_state" not in tool.input_schema["properties"]
    assert result["diagnosis"] == "root cause report"
    assert observed["validation_paths"] == [validation]
    assert saved["agent_state"]["middle_context"]["marker"] == {"complete": True}
    assert saved["extra_trajectory"] == "runtime event stream"


@pytest.mark.asyncio
async def test_agents_have_expected_permissions_and_tools(tmp_path: Path) -> None:
    workspace, _, _ = _workspace(tmp_path)
    diagnostic = diagnostic_agent_module.create_diagnostic_agent(object(), workspace)
    optimizer = optimizer_agent_module.create_optimizer_agent(object(), workspace, diagnostic_model=object())

    diagnostic_names = {schema["function"]["name"] for schema in await diagnostic.toolkit.get_tool_schemas()}
    optimizer_names = {schema["function"]["name"] for schema in await optimizer.toolkit.get_tool_schemas()}

    assert diagnostic.state.permission_context.mode == PermissionMode.EXPLORE
    assert {
        "Read",
        "Grep",
        "Glob",
        "list_validation_results",
        "inspect_git_history",
        "compare_git_versions",
        "list_memory_histories",
        "inspect_memory_history",
        "compare_memory_versions",
        "compare_memory_snapshots",
    } <= diagnostic_names
    assert optimizer.state.permission_context.mode == PermissionMode.DONT_ASK
    assert optimizer.compression_enabled is True
    assert optimizer.context_config.trigger_ratio == 0.8
    assert {
        "Read",
        "Edit",
        "Write",
        "Bash",
        "list_validation_results",
        "list_memory_histories",
        "inspect_memory_history",
        "compare_memory_versions",
        "compare_memory_snapshots",
        "validation_tool",
        "diagnostic_subagent_tool",
    } <= optimizer_names
    assert str(workspace / "code/repo/reme") in optimizer.state.permission_context.working_directories
    assert optimizer.state.permission_context.allow_rules["Read"][0].rule_content == f"{workspace}/**"
    assert optimizer.state.permission_context.allow_rules["Edit"][0].rule_content == f"{workspace}/code/repo/reme/**"
    assert {rule.rule_content for rule in optimizer.state.permission_context.allow_rules["Bash"]} >= {
        f"find {workspace}",
        "git status",
        "git commit",
        "python3 -m json.tool",
        "pytest tests/",
    }


def test_optimizer_can_enable_context_compression_from_its_yaml(tmp_path: Path, monkeypatch) -> None:
    workspace, _, _ = _workspace(tmp_path)
    prompt = utils.load_agent_prompt(META_REME / "as/agent/optimizer_agent.py")
    enabled_prompt = replace(
        prompt,
        context_compression=utils.ContextCompressionConfig(enabled=True, trigger_ratio=0.65, reserve_ratio=0.1),
    )
    monkeypatch.setattr(optimizer_agent_module, "load_agent_prompt", lambda _: enabled_prompt)

    optimizer = optimizer_agent_module.create_optimizer_agent(
        object(),
        workspace,
        diagnostic_model=object(),
    )

    assert optimizer.compression_enabled is True
    assert optimizer.context_config.trigger_ratio == 0.65


def test_bound_validation_tool_uses_explicit_workspace(tmp_path: Path, monkeypatch) -> None:
    workspace, _, _ = _workspace(tmp_path)
    (workspace / "dataset/cases/case.json").write_text(json.dumps({"case_id": "case-1"}), encoding="utf-8")
    calls = []

    def fake_validation(root, case_ids, concurrency, *, validation_id, fail_fast):
        calls.append((Path(root), case_ids, concurrency, fail_fast))
        result = workspace / f"evaluations/init/new/{validation_id}"
        result.mkdir(parents=True)
        (result / "summary.json").write_text(
            json.dumps({"status": "completed", "case_count": 1, "mean_query_score": 1.0, "cases": []}),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(validation_tool_module, "run_validation", fake_validation)
    tool = validation_tool_module.create_validation_tool(workspace, concurrency=3)
    result = tool._func(["case-1"], False)

    assert calls == [(workspace.resolve(), ["case-1"], 3, False)]
    assert result["mean_query_score"] == 1.0
