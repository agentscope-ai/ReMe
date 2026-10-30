"""Single case investigations use normal ReMe lifecycle and sandboxed executors."""

# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,protected-access

import asyncio
import json
import shutil
import sys
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from agentscope.message import TextBlock, ToolCallBlock
from agentscope.model import ChatResponse, ChatUsage
from agentscope.tool import ExecResult
from reme_home.agent import case_study
from reme_home.agent.sandbox import create_case_study_sandbox
from reme_home.evaluation.artifacts import write_json
from reme_home.preparation.code import git

from test_bounded_agent import RecordingModel
from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    home_application,
    preparation_inputs,
    prepared_home,
)
from test_evaluation_inspection import write_run, snapshot

from reme.utils.local_backend import LocalProcessBackend


class RecordingSandbox:
    """Mock only the subprocess boundary; never claim this is an OS sandbox."""

    os_name = "posix"

    def __init__(self, **kwargs):
        self.policy = kwargs
        self.calls = []

    async def exec_shell(self, command, *, cwd, timeout):
        self.calls.append((command, cwd, timeout))
        return ExecResult(exit_code=0, stdout=b"analysis\n", stderr=b"")


@pytest.fixture
def sandbox(monkeypatch):
    instances = []

    def create(**kwargs):
        backend = RecordingSandbox(**kwargs)
        instances.append(backend)
        return backend

    monkeypatch.setattr(case_study, "create_case_study_sandbox", create)
    return instances


@pytest.fixture
def case_run(prepared):
    commit = git(prepared / "code/repo", "rev-parse", "HEAD")
    source = sorted((prepared / "datasets/training-set/cases").glob("*.json"))[0]
    case = json.loads(source.read_text())
    case_id = case["case_id"]
    query_id = case["queries"][0]["query_id"]
    run = write_run(prepared, commit=commit, case_ids=[case_id], query_ids={case_id: [query_id]})
    write_json(
        run / f"cases/{case_id}/queries/result.json",
        {
            "queries": [{"query_id": query_id, "answer": "wrong", "score": 0}],
        },
    )
    return run, case_id, query_id, commit, source


class InvestigationModel(RecordingModel):
    def __init__(self, query_id):
        super().__init__()
        self.query_id = query_id

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        instructions = [
            ("home_trace_query_evidence", {"query_id": self.query_id}),
            ("home_read_code_file", {"path": "plugins/beam/src/reme_beam/__init__.py"}),
            ("python_execute", {"code": "print('analysis')"}),
            ("AutoBash", {"command": "for x in 1 2; do echo $x; done"}),
        ]
        index = len(self.calls) - 1
        if index < len(instructions):
            name, arguments = instructions[index]
            content = [ToolCallBlock(id=f"call-{index}", name=name, input=json.dumps(arguments))]
        else:
            content = [TextBlock(text="Evidence-supported report; root cause remains uncertain.")]
        return ChatResponse(content=content, is_last=True, usage=ChatUsage(input_tokens=2, output_tokens=3, time=0))


@pytest.mark.asyncio
async def test_case_study_uses_dedicated_model_without_changing_judge(app):
    models = app.context.components["as_llm"]
    wrapper = app.context.components["agent_wrapper"]["home_case_study"]
    assert wrapper.as_llm is models["home_case_study"]
    assert wrapper.as_llm.kwargs["model"] == "qwen3.7-max"
    assert app.context.components["agent_wrapper"]["judge"].as_llm is models["judge"]
    assert models["home_case_study"] is not models["judge"]


@pytest.mark.asyncio
async def test_single_run_real_agent_tools_scope_and_no_scratch(app, prepared, case_run, sandbox):
    run, case_id, query_id, commit, source = case_run
    wrapper = app.context.components["agent_wrapper"]["home_case_study"]
    model = InvestigationModel(query_id)
    wrapper.as_llm = SimpleNamespace(model=model)
    (prepared / "code/repo/plugins/beam/src/reme_beam/__init__.py").write_text("value = 999\n")
    before = snapshot(prepared)
    response = await app.run_job(
        "home_case_study",
        validation_path=str(run),
        case_id=case_id,
        focus="Trace the wrong answer",
    )
    assert response.success, response.answer
    assert "Evidence-supported report" in response.answer
    assert response.metadata["candidate_commit"] == commit and response.metadata["usage"]["total_tokens"] == 25
    assert response.metadata["session_id"]
    results = [b for m in model.calls[1]["messages"] for b in m.content if b.type == "tool_result"]
    evidence_result = json.loads(results[0].output[0].text)
    assert evidence_result["locations"]["saved_query"] == "/queries/0"
    assert "value = 1" in str(model.calls[2]["messages"])
    assert "value = 999" not in str(model.calls[2]["messages"])
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in model.calls[0]["tools"]}
    assert set(schemas) == set(case_study.CASE_TOOLS + case_study.CODE_TOOLS) | {"AutoBash", "python_execute"}
    assert "case_id" not in schemas["home_trace_query_evidence"]["properties"]
    assert "validation_path" not in schemas["home_inspect_validation"]["properties"]
    assert "revision" not in schemas["home_read_code_file"]["properties"]
    assert "target_validation_path" not in schemas["home_compare_memory"]["properties"]
    assert "base_ref" in schemas["home_compare_memory"]["required"]
    assert schemas["home_compare_memory"]["properties"]["base_ref"]["type"] == "string"
    descriptions = {item["function"]["name"]: item["function"]["description"] for item in model.calls[0]["tools"]}
    assert "repository-relative" in descriptions["home_read_code_file"]
    assert "python_execute" in descriptions["home_read_code_file"]
    assert "pipefail" in descriptions["AutoBash"]
    assert "do not pass args" in descriptions["python_execute"]
    policy = sandbox[0].policy
    assert policy["writable_paths"] == [] and policy["cwd"] == str(run / f"cases/{case_id}")
    assert str(prepared / "code") in policy["read_only_paths"] and str(source) in policy["read_only_paths"]
    assert str(prepared) not in policy["read_only_paths"]
    assert len(sandbox[0].calls) == 3  # sandbox probe, Python, unrestricted Bash
    assert sandbox[0].calls[1][0][:4] == [sys.executable, "-I", "-B", "-c"]
    assert "for x in 1 2" in str(sandbox[0].calls[2][0])
    assert snapshot(prepared) == before
    # Each invocation is fresh, even through the same application-owned wrapper.
    wrapper.as_llm = SimpleNamespace(model=InvestigationModel(query_id))
    second = await app.run_job("home_case_study", validation_path=str(run), case_id=case_id)
    assert second.success and second.metadata["session_id"] != response.metadata["session_id"]


@pytest.mark.asyncio
async def test_scope_override_rejected_by_real_job_tool(app, prepared, case_run, sandbox):
    run, case_id, _query_id, _commit, _source = case_run
    wrapper = app.context.components["agent_wrapper"]["home_case_study"]
    scope = case_study.prepare_scope(prepared, case_study.CaseStudyInput(validation_path=str(run), case_id=case_id))
    job_parameters = json.dumps(app.context.jobs["home_inspect_memory_history"].parameters, sort_keys=True)
    toolkit = case_study.create_toolkit(wrapper, scope, RecordingSandbox())
    assert json.dumps(app.context.jobs["home_inspect_memory_history"].parameters, sort_keys=True) == job_parameters
    schemas = await toolkit.get_tool_schemas()
    assert len(schemas) == 9
    tool = wrapper._make_tool(
        app.context.jobs["home_inspect_validation"],
        injected_job_kwargs={
            "validation_path": str(run),
            "case_id": case_id,
        },
    )
    with pytest.raises(ValueError, match="cannot be provided"):
        await tool._func(case_id="another-case")


@pytest.mark.asyncio
async def test_invalid_scope_and_sandbox_failure_precede_llm(app, prepared, case_run, sandbox, monkeypatch):
    run, case_id, _query_id, _commit, _source = case_run
    wrapper = app.context.components["agent_wrapper"]["home_case_study"]
    model = InvestigationModel("unused")
    wrapper.as_llm = SimpleNamespace(model=model)
    response = await app.run_job("home_case_study", validation_path=str(run), case_id="../escape")
    assert not response.success and not sandbox
    response = await app.run_job(
        "home_case_study",
        validation_path=str(run),
        case_id=case_id,
        _allowed_paths=[str(run)],
    )
    assert not response.success and not sandbox

    async def denied(self, *args, **kwargs):
        return ExecResult(exit_code=1, stdout=b"", stderr=b"sandbox denied")

    monkeypatch.setattr(RecordingSandbox, "exec_shell", denied)
    response = await app.run_job("home_case_study", validation_path=str(run), case_id=case_id)
    assert not response.success and "sandbox unavailable" in str(response.answer)
    assert not model.calls

    def unavailable(**kwargs):
        raise RuntimeError("No sandbox backend")

    monkeypatch.setattr(case_study, "create_case_study_sandbox", unavailable)
    response = await app.run_job("home_case_study", validation_path=str(run), case_id=case_id)
    assert not response.success and "No sandbox backend" in str(response.answer) and not model.calls


@pytest.mark.asyncio
async def test_iteration_limit_returns_final_report(app, prepared, case_run, sandbox):
    run, case_id, _query_id, _commit, _source = case_run
    wrapper = app.context.components["agent_wrapper"]["home_case_study"]
    assert wrapper.kwargs["react_config"]["max_iters"] == 60
    assert wrapper.kwargs["time_budget_minutes"] == 30
    wrapper.kwargs["react_config"] = {"max_iters": 1}
    model = RecordingModel()
    wrapper.as_llm = SimpleNamespace(model=model)
    response = await app.run_job("home_case_study", validation_path=str(run), case_id=case_id)
    assert response.success and response.answer == "final report"
    assert model.calls[-1]["tools"] == []


@pytest.mark.asyncio
async def test_os_sandbox_read_scope_and_no_writes(tmp_path):
    case = tmp_path / "case"
    code = tmp_path / "code"
    case.mkdir()
    code.mkdir()
    (code / "source.py").write_text("visible code")
    git(code, "init")
    git(code, "add", "source.py")
    git(code, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial")
    outside = tmp_path / "secret.txt"
    outside.write_text("outside scope")
    (case / "manifest.json").write_text("{}")
    backend = create_case_study_sandbox(
        cwd=str(case),
        read_only_paths=[str(case), str(code)],
        writable_paths=[],
        memory_limit_mib=0,
    )
    try:
        probe = await backend.exec_shell([sys.executable, "-I", "-B", "-c", "pass"], cwd=str(case), timeout=10)
    except (OSError, RuntimeError) as error:
        pytest.skip(f"OS sandbox unavailable on this host: {error}")
    if not probe.ok():
        pytest.skip("OS sandbox cannot start on this host: " + probe.stderr.decode(errors="replace"))
    ready = await case_study.probe_sandbox(
        backend,
        {
            "case_path": str(case),
            "source_case": str(code / "source.py"),
            "validation_path": str(case),
            "code_repository": str(code),
            "candidate_commit": git(code, "rev-parse", "HEAD"),
        },
    )
    assert ready.ok(), ready.stderr.decode(errors="replace")
    tool = case_study.python_tool(backend, str(case))
    result = await tool._func(code=f"from pathlib import Path; print(Path({str(code / 'source.py')!r}).read_text())")
    assert result["returncode"] == 0 and "visible code" in result["stdout"]
    result = await tool._func(
        code=f"import subprocess; subprocess.run(['git', '-C', {str(code)!r}, 'show', 'HEAD:source.py'], check=True)",
    )
    assert result["returncode"] == 0 and "visible code" in result["stdout"], result
    for script in (
        f"open({str(outside)!r}).read()",
        "open('written.txt', 'w').write('denied')",
        f"open({str(code / 'source.py')!r}, 'w').write('changed')",
    ):
        result = await tool._func(code=script)
        assert result["returncode"] != 0
    bash = case_study.AutoBash(cwd=str(case), backend=backend, is_read_only=False)
    chunks = [chunk async for chunk in bash.call(f"cat '{code / 'source.py'}'")]
    assert "visible code" in str(chunks)
    chunks = [chunk async for chunk in bash.call("echo changed > written.txt")]
    assert "Command failed" in str(chunks)
    assert not (case / "written.txt").exists() and (code / "source.py").read_text() == "visible code"
    chunks = [chunk async for chunk in bash.call(f"git -C '{code}' show HEAD:source.py")]
    assert "visible code" in str(chunks), chunks
    git_config = (code / ".git/config").read_bytes()
    result = await tool._func(
        code=(
            f"import subprocess; subprocess.run(['git', '-C', {str(code)!r},"
            " 'config', 'test.blocked', 'yes'], check=True)"
        ),
    )
    assert result["returncode"] != 0
    assert (code / ".git/config").read_bytes() == git_config


@pytest.mark.asyncio
async def test_python_code_file_timeout_and_cancellation(tmp_path):
    # Execution protocol uses real subprocesses; OS enforcement is tested separately.
    tool = case_study.python_tool(LocalProcessBackend(), str(tmp_path))
    result = await tool._func(code="print(1 + 2)")
    assert result["stdout"].strip() == "3" and result["returncode"] == 0
    script = tmp_path / "script.py"
    script.write_text("import sys; print(sys.argv[1])")
    result = await tool._func(file="script.py", args=["argument"])
    assert result["stdout"].strip() == "argument"
    result = await tool._func(code="import time; time.sleep(20)", timeout=0.05)
    assert result["timed_out"]
    task = asyncio.create_task(tool._func(code="import time; time.sleep(20)"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for arguments in (
        {},
        {"code": "pass", "file": "x"},
        {"code": "pass", "args": []},
        {"code": "pass", "timeout": float("nan")},
        {"code": "pass", "timeout": True},
    ):
        with pytest.raises(ValueError):
            await tool._func(**arguments)


@pytest.fixture
def batch_run(case_run):
    run, case_id, query_id, commit, source = case_run
    case = json.loads(source.read_text())
    case["case_id"] = "case2"
    write_json(source.parent / "batch-second.json", case)
    shutil.copytree(run / f"cases/{case_id}", run / "cases/case2")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["case_ids"].append("case2")
    manifest["query_ids"]["case2"] = [query_id]
    write_json(run / "manifest.json", manifest)
    return run, case_id, query_id, commit


class BatchModel(RecordingModel):
    """Read each fresh agent's case-local evidence before returning its report."""

    def __init__(self, query_id):
        super().__init__()
        self.query_id = query_id

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        results = [b for m in kwargs["messages"] for b in m.content if b.type == "tool_result"]
        if not results:
            content = [
                ToolCallBlock(
                    id="batch-trace",
                    name="home_trace_query_evidence",
                    input=json.dumps({"query_id": self.query_id}),
                ),
            ]
        else:
            content = [TextBlock(text="case-local report")]
        return ChatResponse(content=content, is_last=True, usage=ChatUsage(input_tokens=2, output_tokens=3, time=0))


@pytest.mark.asyncio
async def test_batch_shares_focus_but_isolates_agents_tools_and_sandboxes(app, prepared, batch_run, sandbox):
    run, first, query_id, commit = batch_run
    model = BatchModel(query_id)
    app.context.components["agent_wrapper"]["home_case_study"].as_llm = SimpleNamespace(model=model)
    before = snapshot(prepared)
    response = await app.run_job(
        "home_case_study",
        evaluation_path=str(run),
        case_ids=["case2", first],
        focus="Shared hypothesis",
    )
    assert response.success, response.answer
    results = response.answer["results"]
    assert [item["case_id"] for item in results] == ["case2", first]
    assert all(item["report"] == "case-local report" for item in results)
    assert len({item["session_id"] for item in results}) == 2
    assert response.metadata["evaluation_path"] == str(run) and response.metadata["candidate_commit"] == commit
    assert response.metadata["usage"]["total_tokens"] == 20
    assert len(model.calls) == 4
    for i, case_id in enumerate(["case2", first]):
        calls = model.calls[i * 2 : i * 2 + 2]
        assert "Shared hypothesis" in str(calls[0]["messages"])
        outputs = [b for m in calls[1]["messages"] for b in m.content if b.type == "tool_result"]
        evidence = json.loads(outputs[0].output[0].text)
        assert evidence["paths"]["query_result"] == str(run / f"cases/{case_id}/queries/result.json")
        assert sandbox[i].policy["cwd"] == str(run / f"cases/{case_id}")
        other = first if case_id == "case2" else "case2"
        assert str(run / f"cases/{other}") not in sandbox[i].policy["read_only_paths"]
    assert snapshot(prepared) == before


@pytest.mark.asyncio
async def test_batch_continues_after_case_failure_and_keeps_usage_unknown(app, batch_run, sandbox, monkeypatch):
    run, first, query_id, _commit = batch_run
    model = BatchModel(query_id)
    app.context.components["agent_wrapper"]["home_case_study"].as_llm = SimpleNamespace(model=model)
    original = RecordingSandbox.exec_shell

    async def fail_second_case(self, command, *, cwd, timeout):
        if str(cwd).endswith("/case2"):
            return ExecResult(exit_code=1, stdout=b"", stderr=b"sandbox denied")
        return await original(self, command, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(RecordingSandbox, "exec_shell", fail_second_case)
    response = await app.run_job("home_case_study", validation_path=str(run), case_ids=["case2", first])
    assert not response.success
    failed, succeeded = response.answer["results"]
    assert failed["case_id"] == "case2" and "sandbox unavailable" in failed["error"]
    assert succeeded["case_id"] == first and succeeded["success"] and succeeded["session_id"]
    assert response.metadata["usage"] is None


@pytest.mark.asyncio
async def test_entire_batch_is_validated_before_spending_model_tokens(app, batch_run, sandbox):
    run, first, _query_id, _commit = batch_run
    response = await app.run_job("home_case_study", evaluation_path=str(run), case_ids=[first, "missing"])
    assert not response.success and "not selected" in response.answer and not sandbox


@pytest.mark.asyncio
async def test_batch_cancellation_propagates_without_starting_next_case(app, batch_run, sandbox, monkeypatch):
    run, first, _query_id, _commit = batch_run

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(app.context.components["agent_wrapper"]["home_case_study"], "reply", cancel)
    with pytest.raises(asyncio.CancelledError):
        await app.run_job("home_case_study", evaluation_path=str(run), case_ids=[first, "case2"])
    assert len(sandbox) == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"case_ids": []},
        {"case_ids": ["case1", "case1"]},
        {"case_ids": ["../escape"]},
        {"case_ids": [str(i) for i in range(9)]},
        {"case_ids": "case1"},
        {"case_ids": [1]},
        {"validation_path": "another"},
        {"case_id": "case1"},
    ],
)
def test_batch_request_rejects_invalid_or_ambiguous_inputs(updates):
    with pytest.raises(ValidationError):
        case_study.CaseStudyBatchInput.model_validate(
            {"evaluation_path": "evaluations/a/b/1", "case_ids": ["case1"], **updates},
        )


@pytest.mark.parametrize("path_key", ["evaluation_path", "validation_path"])
@pytest.mark.parametrize("case_key", ["case_ids", "case_id"])
@pytest.mark.asyncio
async def test_new_and_legacy_selectors_match_job_schema(app, batch_run, sandbox, path_key, case_key):
    run, first, query_id, _commit = batch_run
    app.context.components["agent_wrapper"]["home_case_study"].as_llm = SimpleNamespace(model=BatchModel(query_id))
    arguments = {path_key: str(run), case_key: [first] if case_key == "case_ids" else first}
    schema = app.context.jobs["home_case_study"].parameters
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(arguments)
    assert not validator.is_valid({**arguments, "evaluation_path": str(run), "validation_path": str(run)})
    assert not validator.is_valid({**arguments, "case_ids": [first], "case_id": first})
    response = await app.run_job("home_case_study", **arguments)
    assert response.success, response.answer
    if case_key == "case_id":
        assert response.answer == "case-local report" and response.metadata["case_id"] == first
    else:
        assert response.answer["results"][0]["case_id"] == first
