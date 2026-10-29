"""Single case investigations use normal ReMe lifecycle and sandboxed executors."""

# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,protected-access

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
from agentscope.message import TextBlock, ToolCallBlock
from agentscope.model import ChatResponse, ChatUsage
from agentscope.tool import ExecResult
from reme_home.agent import case_study
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
from reme.utils.file_sandbox import create_file_sandbox_backend


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

    monkeypatch.setattr(case_study, "create_file_sandbox_backend", create)
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
    assert "value = 1" in str(model.calls[2]["messages"])
    assert "value = 999" not in str(model.calls[2]["messages"])
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in model.calls[0]["tools"]}
    assert set(schemas) == set(case_study.CASE_TOOLS + case_study.CODE_TOOLS) | {"AutoBash", "python_execute"}
    assert "case_id" not in schemas["home_trace_query_evidence"]["properties"]
    assert "validation_path" not in schemas["home_inspect_validation"]["properties"]
    assert "revision" not in schemas["home_read_code_file"]["properties"]
    assert "target_validation_path" not in schemas["home_compare_memory"]["properties"]
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
    toolkit = case_study.create_toolkit(wrapper, scope, RecordingSandbox())
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

    monkeypatch.setattr(case_study, "create_file_sandbox_backend", unavailable)
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
    outside = tmp_path / "secret.txt"
    outside.write_text("outside scope")
    backend = create_file_sandbox_backend(
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
    tool = case_study.python_tool(backend, str(case))
    result = await tool._func(code=f"from pathlib import Path; print(Path({str(code / 'source.py')!r}).read_text())")
    assert result["returncode"] == 0 and "visible code" in result["stdout"]
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
