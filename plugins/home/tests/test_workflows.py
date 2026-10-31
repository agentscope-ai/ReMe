"""Minimal Home agents exercise real Jobs, Git and AS tools with fake model/Docker boundaries."""

# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,protected-access

import asyncio
import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import HintBlock, TextBlock, ToolCallBlock
from agentscope.model import ChatResponse, ChatUsage
from agentscope.tool import LocalBackend
from jsonschema import Draft202012Validator
from reme_home.agent import optimizer as optimizer_module, diagnostic as diagnostic_module, workflow_common, case_study
from reme_home.evaluation.artifacts import read_json, write_json
from reme_home.preparation.code import git
from reme_home.workspace import _initialization_lock

from test_bounded_agent import RecordingModel
from test_case_study import RecordingSandbox
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    home_application,
    preparation_inputs,
    prepared_home,
)


class ScriptedModel(RecordingModel):
    def __init__(self, calls=(), final="Supported diagnostic report."):
        super().__init__()
        self.script = list(calls)
        self.report = final

    async def __call__(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if index < len(self.script):
            name, arguments = self.script[index]
            arguments = arguments() if callable(arguments) else arguments
            content = [ToolCallBlock(id=f"call-{index}", name=name, input=json.dumps(arguments))]
        else:
            final = self.report() if callable(self.report) else self.report
            content = [TextBlock(text=final)]
        return ChatResponse(content=content, is_last=True, usage=ChatUsage(input_tokens=2, output_tokens=3, time=0))


@pytest.fixture
def sandbox(monkeypatch):
    policies = []

    def create(**kwargs):
        # Mock only subprocess isolation. Writes are restricted by the real
        # sandbox in the dedicated OS tests below, not by this local fake.
        policies.append(kwargs)
        local = LocalBackend()
        return SimpleNamespace(backend=local, exec_shell=local.exec_shell)

    monkeypatch.setattr(workflow_common, "create_case_study_sandbox", create)
    monkeypatch.setattr(case_study, "create_case_study_sandbox", RecordingSandbox)
    return policies


def set_model(app, name, model):
    wrapper = app.context.components["agent_wrapper"][name]
    wrapper.as_llm = SimpleNamespace(model=model)
    return wrapper


async def baseline(app):
    result = await app.run_job("home_evaluate")
    assert result.success, result.answer
    return result.answer["result_dir"]


def final_report(home, branch="init", path=None):
    repo = home / "code/repo"
    if path is None:
        path = sorted((home / "evaluations" / branch).glob("*/*/summary.json"))[-1].parent
    return json.dumps(
        {
            "branch_name": branch,
            "commit_sha": git(repo, "rev-parse", "HEAD"),
            "evaluation_path": str(path),
            "report": "Selected using matching formal evidence; sampled coverage limits confidence.",
        },
    )


@pytest.mark.asyncio
async def test_real_optimization_diagnoses_edits_debugs_and_evaluates(app, prepared, sandbox, monkeypatch):
    home = prepared
    now, timers = [0.0], []
    timer_class = optimizer_module.ValidationCycleTimer

    def create_timer(budget):
        timer = timer_class(budget, clock=lambda: now[0])
        timers.append(timer)
        return timer

    class TimedModel(ScriptedModel):
        async def __call__(self, **kwargs):
            formatted = await OpenAIChatFormatter().format(kwargs["messages"])
            pending = set()
            for message in formatted:
                if pending:
                    assert message["role"] == "tool", formatted
                    pending.remove(message["tool_call_id"])
                else:
                    assert message["role"] != "tool", formatted
                    pending.update(call["id"] for call in message.get("tool_calls", []))
            assert not pending
            response = await super().__call__(**kwargs)
            now[0] += 60 * 60
            return response

    monkeypatch.setattr(optimizer_module, "ValidationCycleTimer", create_timer)
    target = home / "code/repo/plugins/beam/src/reme_beam/__init__.py"
    baseline_path = await baseline(app)
    diagnostic = ScriptedModel(
        [("home_case_study", {"evaluation_path": baseline_path, "case_ids": ["1"], "focus": "Trace wrong answers"})],
    )
    set_model(app, "home_diagnostic", diagnostic)
    set_model(app, "home_case_study", ScriptedModel())
    optimizer = TimedModel(
        [
            ("get_validation_budget", {}),
            ("home_diagnose", {"evaluation_paths": [baseline_path]}),
            ("home_create_candidate_branch", {"branch_name": "mechanism"}),
            ("Read", {"file_path": str(target)}),
            ("Edit", {"file_path": str(target), "old_string": "value = 1", "new_string": "value = 2"}),
            ("home_commit_candidate", {"message": "Change mechanism"}),
            ("home_evaluate", {"debug": True}),
            ("home_evaluate", {"debug": False}),
        ],
        final=lambda: final_report(home, "mechanism"),
    )
    set_model(app, "home_optimizer", optimizer)
    response = await app.run_job("home_optimize", instruction="Improve general retrieval")
    assert response.success, response.answer
    assert target.read_text() == "value = 2\n"
    assert response.answer["branch_name"] == "mechanism"
    assert response.answer["status"] == "completed"
    transitions = [
        block.hint
        for message in optimizer.calls[-1]["messages"]
        for block in message.content
        if isinstance(block, HintBlock) and block.source == "meta_reme_validation_cycle_transition"
    ]
    assert len(transitions) == 1
    assert "Formal validation 1 completed successfully" in transitions[0]
    assert "Remaining formal validations: 9. Start cycle 2." in transitions[0]
    assert len(timers) == 1
    assert timers[0]._started_at == now[0] - 60 * 60  # Reset before the final reasoning call.
    assert any(
        isinstance(block, HintBlock) and block.source == "meta_reme_validation_timer"
        for message in optimizer.calls[-1]["messages"]
        for block in message.content
    )
    assert response.metadata["usage"]["input_tokens"] == 2 * len(optimizer.calls)
    assert response.metadata["usage"]["output_tokens"] == 3 * len(optimizer.calls)
    assert "sampled coverage" in response.answer["report"]
    assert read_json(Path(response.metadata["optimization_report_path"]))["status"] == "completed"
    assert len(diagnostic.calls) == 2
    source = next((home / "logs/diagnostic-trajectories").glob("*.json"))
    trace = read_json(source)
    assert trace["session_id"] and any("Improve general retrieval" in str(msg) for msg in trace["messages"])
    assert "permission_context" not in trace
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in optimizer.calls[0]["tools"]}
    assert set(schemas["home_evaluate"]["properties"]) == {"debug"}
    assert "caller_trajectory" not in schemas["home_diagnose"]["properties"]
    assert "get_validation_budget" in schemas
    assert not any("equivalence" in name for name in schemas)
    for manifest_path in (home / "evaluations").glob("*/*/*/manifest.json"):
        manifest = read_json(manifest_path)
        assert manifest["execution_mode"] == "full_validation"
    assert any(read_json(path)["mode"] == "debug" for path in (home / "evaluations").glob("*/*/*/manifest.json"))
    diagnostic_schemas = {item["function"]["name"] for item in diagnostic.calls[0]["tools"]}
    assert "home_case_study" in diagnostic_schemas
    assert not set(optimizer_module.MUTATION_JOBS) & diagnostic_schemas
    assert "Edit" not in diagnostic_schemas and "home_evaluate" not in diagnostic_schemas
    assert "Write" in diagnostic_schemas  # Private scratch only.
    assert any(str(target.parent.parent) in policy["writable_paths"] for policy in sandbox)
    assert all(str(home / "code/repo") not in policy["writable_paths"] for policy in sandbox)


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["beam", "longmemeval"])
async def test_benchmark_optimizer_applies_budgets_to_real_workflow(
    app,
    prepared,
    inputs,
    sandbox,
    monkeypatch,
    tmp_path,
    benchmark,
):
    baseline_path = await baseline(app)
    await app.close()
    module = import_module(f"benchmark.{benchmark}.run_home")
    config = tmp_path / "home.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(prepared.parent),
                "preparation": inputs[0],
                "max_validations": 3,
                "validation_cycle_time_budget_minutes": 30,
            },
        ),
    )
    application_class = type(app)
    agents = []
    model = ScriptedModel([("get_validation_budget", {})], final=lambda: final_report(prepared, path=baseline_path))

    def factory(**kwargs):
        application = application_class(**kwargs, log_to_console=False)
        original_start = application.start

        async def start():
            await original_start()
            wrapper = set_model(application, "home_optimizer", model)
            original_create = wrapper._create_agent

            def create(**options):
                agent = original_create(**options)
                agents.append(agent)
                return agent

            monkeypatch.setattr(wrapper, "_create_agent", create)

        monkeypatch.setattr(application, "start", start)
        return application

    monkeypatch.setattr(module, "Application", factory)
    result = await module.optimize(config, instruction="Keep the best existing candidate")
    assert result["status"] == "completed"
    assert result["branch_name"] == "init"
    assert len(agents) == 1
    assert agents[0].validation_cycle_timer.time_budget_minutes == 30
    task = "\n".join(msg.get_text_content() or "" for msg in model.calls[0]["messages"])
    assert '"max_validations": 3' in task
    assert '"remaining_validations": 3' in task
    assert len(FakeFactory.instances) == 1  # Only the earlier baseline ran.


@pytest.mark.asyncio
async def test_optimizer_creates_baseline_once_and_verifies_baseline_fallback(app, prepared, sandbox):
    model = ScriptedModel(final=lambda: final_report(prepared))
    set_model(app, "home_optimizer", model)
    response = await app.run_job("home_optimize")
    assert response.success, response.answer
    assert len(FakeFactory.instances) == 1
    assert response.answer["branch_name"] == "init"
    run = response.answer["baseline_path"]
    manifest = read_json(Path(run) / "manifest.json")
    assert set(manifest["case_ids"]) == {"1", "2"} and not manifest["options"]["sampling"]
    assert manifest["options"]["record_history"]
    set_model(app, "home_optimizer", ScriptedModel(final=lambda: final_report(prepared)))
    second = await app.run_job("home_optimize")
    assert second.success, second.answer
    assert len(FakeFactory.instances) == 1
    assert second.answer["session_id"] != response.answer["session_id"]


@pytest.mark.asyncio
async def test_standalone_diagnosis_fresh_state_and_persisted_provenance(app, prepared, sandbox):
    path = await baseline(app)
    model = ScriptedModel()
    set_model(app, "home_diagnostic", model)
    first = await app.run_job("home_diagnose", evaluation_paths=[path])
    second = await app.run_job("home_diagnose", evaluation_paths=[path], instruction="Compare failure clusters")
    assert first.success and second.success
    assert first.answer["session_id"] != second.answer["session_id"]
    assert first.answer["trajectory_path"] != second.answer["trajectory_path"]
    assert first.answer["evaluation_paths"] == [path]
    record = read_json(Path(first.answer["diagnostic_report_path"]))
    assert record["diagnosis"] == "Supported diagnostic report."
    assert read_json(Path(first.answer["trajectory_path"])) == {"messages": [], "summary": None}
    assert all(
        len(policy["writable_paths"]) == 1 and "/tmp/diagnostic-" in policy["writable_paths"][0] for policy in sandbox
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["infra", "partial", "debug", "replay", "wrong_commit"])
async def test_bad_baseline_rejected_before_optimizer(app, prepared, sandbox, change):
    path = Path(await baseline(app))
    manifest = read_json(path / "manifest.json")
    summary = read_json(path / "summary.json")
    if change == "infra":
        summary["case_status_counts"]["infra_error"] = 1
    elif change == "partial":
        manifest["case_ids"] = ["1"]
        manifest["query_ids"] = {"1": manifest["query_ids"]["1"]}
    elif change == "debug":
        manifest["mode"] = "debug"
    elif change == "replay":
        manifest["execution_mode"] = "query_only_replay"
    else:
        manifest["commit_sha"] = "0" * 40
    write_json(path / "manifest.json", manifest)
    write_json(path / "summary.json", summary)
    # Missing valid baseline must not silently switch away from a candidate.
    response = await app.run_job("home_create_candidate_branch", branch_name="existing")
    assert response.success
    model = ScriptedModel()
    set_model(app, "home_optimizer", model)
    response = await app.run_job("home_optimize")
    assert not response.success and "baseline is missing" in str(response.answer)
    assert not model.calls and not sandbox
    assert git(prepared / "code/repo", "branch", "--show-current") == "existing"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["dirty", "branch", "commit", "path", "infra", "debug", "replay", "invalid_json"])
async def test_final_claim_rejected_without_discarding_changes(app, prepared, sandbox, change):
    path = Path(await baseline(app))
    target = prepared / "code/repo/plugins/beam/src/reme_beam/__init__.py"

    def finish():
        claim = json.loads(final_report(prepared, path=path))
        if change == "dirty":
            target.write_text("preserve unfinished changes\n")
        elif change == "branch":
            claim["branch_name"] = "invented"
        elif change == "commit":
            claim["commit_sha"] = "0" * 40
        elif change == "path":
            claim["evaluation_path"] = str(prepared / "tmp")
        elif change == "infra":
            summary = read_json(path / "summary.json")
            summary["query_status_counts"]["infra_error"] = 1
            write_json(path / "summary.json", summary)
        elif change in {"debug", "replay"}:
            manifest = read_json(path / "manifest.json")
            manifest["mode" if change == "debug" else "execution_mode"] = (
                "debug" if change == "debug" else "query_only_replay"
            )
            write_json(path / "manifest.json", manifest)
        else:
            return "not a structured final claim"
        return json.dumps(claim)

    set_model(app, "home_optimizer", ScriptedModel(final=finish))
    response = await app.run_job("home_optimize")
    assert not response.success
    record = read_json(Path(response.metadata["optimization_report_path"]))
    assert record["status"] == "failed" and record["error"]
    if change == "dirty":
        assert target.read_text() == "preserve unfinished changes\n"


@pytest.mark.asyncio
async def test_infrastructure_baseline_failure_never_calls_optimizer(app, prepared, sandbox, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("Docker unavailable")

    monkeypatch.setattr(FakeFactory, "create_cases", broken)
    model = ScriptedModel()
    set_model(app, "home_optimizer", model)
    response = await app.run_job("home_optimize")
    assert not response.success and not model.calls
    assert "baseline failed" in str(response.answer)


@pytest.mark.asyncio
async def test_invalid_scope_schema_and_injected_trajectory(app, prepared, sandbox):
    path = await baseline(app)
    wrapper = set_model(app, "home_optimizer", ScriptedModel())
    tool = diagnostic_module.diagnostic_tool(wrapper)
    with pytest.raises(ValueError, match="host-owned"):
        await tool._func(evaluation_paths=[path], caller_trajectory={"spoof": True})
    for values in ([str(prepared / "tmp")], [path, path], [], ["../escape"]):
        result = await app.run_job("home_diagnose", evaluation_paths=values)
        assert not result.success
    result = await app.run_job("home_diagnose", evaluation_paths=[path], _allowed_paths=[path])
    assert not result.success and not sandbox
    for name in ("home_optimize", "home_diagnose"):
        Draft202012Validator.check_schema(app.context.jobs[name].parameters)
    assert "caller_trajectory" not in app.context.jobs["home_diagnose"].parameters["properties"]
    with pytest.raises(ValueError, match="execution settings"):
        optimizer_module.evaluation_options({"reuse_construction_from": path})


@pytest.mark.asyncio
async def test_single_optimizer_lock_and_cancellation_cleanup(app, prepared, sandbox, monkeypatch):
    await baseline(app)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()

    monkeypatch.setattr(optimizer_module, "workflow_reply", blocked)
    running = asyncio.create_task(app.run_job("home_optimize"))
    await asyncio.wait_for(entered.wait(), 5)
    second = await app.run_job("home_optimize")
    assert not second.success and "already in progress" in str(second.answer)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    reports = list((prepared / "logs/optimization-reports").glob("*.json"))
    assert len(reports) == 1 and read_json(reports[0])["status"] == "cancelled"
    with _initialization_lock(prepared / "metadata/optimization.lock"):
        pass


@pytest.mark.asyncio
async def test_metadata_link_rejected_before_opening_any_lock(app, prepared, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    state = prepared / "metadata"
    saved = prepared / "saved-metadata"
    state.rename(saved)
    state.symlink_to(outside, target_is_directory=True)
    try:
        result = await app.run_job("home_optimize")
        assert not result.success and "real directory" in str(result.answer)
        with pytest.raises(ValueError, match="real directory"):
            optimizer_module.baseline_state(prepared)
        with pytest.raises(ValueError, match="real directory"):
            optimizer_module.verify_final(prepared, None)
        assert not list(outside.iterdir())
    finally:
        state.unlink()
        saved.rename(state)


@pytest.mark.asyncio
async def test_workflow_sandbox_enforces_read_write_and_git_boundaries(tmp_path):
    home = tmp_path / "home"
    repo = home / "repo"
    candidate = repo / "src"
    scratch = home / "scratch"
    evidence = home / "evidence"
    for directory in (candidate, repo / ".git", scratch, evidence):
        directory.mkdir(parents=True, exist_ok=True)
    target = candidate / "module.py"
    target.write_text("old\n")
    secret = tmp_path / "outside.txt"
    secret.write_text("outside")
    (evidence / "result.json").write_text("evidence")
    for role in ("optimizer", "diagnostic"):
        scope = {
            "scratch": str(scratch),
            "read_only_paths": [str(repo), str(evidence)],
            "writable_paths": [str(scratch), *([str(candidate)] if role == "optimizer" else [])],
        }
        backend = workflow_common.WorkflowSandbox(scope, memory_limit_mib=0)
        assert (await backend.read_file(str(target))) in (b"old\n", b"new\n")
        assert (await backend.stat(str(target))).size_bytes > 0
        assert "module.py" in await backend.list_dir(str(candidate))
        await backend.write_file(str(scratch / "analysis.py"), b"print('analysis')\n")
        assert (await backend.exec_shell([sys.executable, "-I", "-B", str(scratch / "analysis.py")])).ok()
        for path in (repo / ".git/config", evidence / "result.json", secret):
            with pytest.raises(OSError):
                await backend.write_file(str(path), b"forbidden")
        with pytest.raises(OSError):
            await backend.read_file(str(secret))
        if role == "optimizer":
            await backend.write_file(str(target), b"new\n")
            # Replaced roots after managed Git operations must not retain stale inodes.
            replacement = candidate / "replacement"
            replacement.write_text("new\n")
            replacement.replace(target)
            assert await backend.read_file(str(target)) == b"new\n"
        else:
            with pytest.raises(OSError):
                await backend.write_file(str(target), b"forbidden")
    assert secret.read_text() == "outside" and (evidence / "result.json").read_text() == "evidence"
    # A single-file candidate root must survive a managed checkout replacing
    # its inode without accidentally granting writes to its containing directory.
    file_scope = {
        "scratch": str(scratch),
        "read_only_paths": [str(repo)],
        "writable_paths": [str(scratch), str(target)],
    }
    backend = workflow_common.WorkflowSandbox(file_scope, memory_limit_mib=0)
    await backend.write_file(str(target), b"file root\n")
    replacement = candidate / "replacement"
    replacement.write_text("replaced root\n")
    replacement.replace(target)
    assert await backend.read_file(str(target)) == b"replaced root\n"
    with pytest.raises(OSError):
        await backend.write_file(str(candidate / "sibling.py"), b"forbidden")
    target.unlink()
    target.symlink_to(secret)
    with pytest.raises(ValueError, match="symlinks"):
        await backend.read_file(str(target))
