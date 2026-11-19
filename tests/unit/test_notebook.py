"""Todo/Idea job contracts, authoritative JSON and interrupted archive recovery."""

# pylint: disable=protected-access,redefined-outer-name

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from fastmcp import Client
import httpx
from pydantic import ValidationError
import pytest
import pytest_asyncio

from reme.components import ApplicationContext
from reme.components.client.http_client import HttpClient
from reme.components.client.mcp_client import MCPClient
from reme.components.job import BaseJob
from reme.components.service import HttpService
from reme.config import resolve_app_config
from reme.schema import Idea, IdeaRequest, Todo, TodoRequest
from reme.steps.notebook import TodoStep
from reme.steps.notebook import _store


@pytest_asyncio.fixture
async def jobs(tmp_path):
    """Run real registered steps with no models, network or repository workspace."""
    config = resolve_app_config(config="proactive", log_config=False)
    context = ApplicationContext(workspace_dir=str(tmp_path))
    result = {name: BaseJob(name=name, app_context=context, **config["jobs"][name]) for name in ("todo", "idea")}
    context.jobs = result
    for job in result.values():
        await job.start()
    yield result
    for job in reversed(list(result.values())):
        await job.close()


async def call(jobs, name, **kwargs):
    """Require successful business responses and return their JSON-ready payloads."""
    response = await jobs[name](**kwargs)
    assert response.success, response.answer
    return response.answer


def read(path):
    """Inspect durable arrays independently of the management jobs."""
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_todo_progress_dates_archive_and_reopen(jobs, tmp_path):
    """State transitions retain identity, deadlines and the complete progress history."""
    idea = await call(jobs, "idea", action="create", project="ReMe", text="通过对话管理待办")
    task = await call(jobs, "todo", action="create", source_idea=idea["id"], due_date="2026-11-20")
    identifier = task["id"]
    assert task["project"] == "ReMe" and task["text"] == idea["text"]
    assert task["status"] == "open" and task["progress"] == []
    await call(jobs, "todo", action="update", id=identifier, status="partial", progress_text="接口完成，测试待补")
    task = await call(
        jobs,
        "todo",
        action="update",
        id=identifier,
        progress_text="补充边界测试",
        progress_time="2026-11-18T11:00:00+08:00",
    )
    assert task["due_date"] == "2026-11-20" and len(task["progress"]) == 2
    assert task["progress"][1]["time"] == "2026-11-18T11:00:00+08:00"
    assert datetime.fromisoformat(task["progress"][0]["time"]).utcoffset() is not None
    listing = await call(jobs, "todo", action="list")
    assert listing["items"][0]["progress_count"] == 2
    assert len(listing["items"][0]["progress"]) == 1
    closed = await call(jobs, "todo", action="update", id=identifier, status="done")
    assert closed["closed_at"] and read(tmp_path / "todos.json") == []
    assert read(tmp_path / "archive/todos.json") == [closed]
    assert (await call(jobs, "todo", action="list"))["total"] == 0
    assert (await call(jobs, "todo", action="list", scope="archive"))["total"] == 1
    assert await call(jobs, "todo", action="get", id=identifier) == closed
    assert "closed_at" not in await call(jobs, "idea", action="get", id=idea["id"])
    reopened = await call(jobs, "todo", action="update", id=identifier, status="open", due_date=None)
    assert "closed_at" not in reopened and "due_date" not in reopened
    assert reopened["progress"] == closed["progress"] and reopened["id"] == identifier
    assert read(tmp_path / "archive/todos.json") == []
    cancelled = await call(jobs, "todo", action="update", id=identifier, status="cancelled")
    assert cancelled["closed_at"] and cancelled["status"] == "cancelled"


@pytest.mark.asyncio
async def test_existing_links_idea_closure_and_restore(jobs, tmp_path):
    """Links survive independent idea closure without changing existing task content."""
    idea = await call(jobs, "idea", action="create", project="ReMe", text="想法")
    task = await call(jobs, "todo", action="create", project="Other", text="已有任务")
    linked = await call(jobs, "todo", action="update", id=task["id"], source_idea=idea["id"])
    assert linked["project"] == "Other" and linked["text"] == "已有任务" and linked["status"] == "open"
    await call(jobs, "idea", action="update", id=idea["id"], text="补充想法")
    closed = await call(jobs, "idea", action="archive", id=idea["id"])
    assert read(tmp_path / "ideas.json") == []
    assert read(tmp_path / "archive/ideas.json") == [closed]
    inherited = await call(jobs, "todo", action="create", source_idea=idea["id"])
    assert inherited["text"] == "补充想法"
    assert (await call(jobs, "todo", action="list", source_idea=idea["id"]))["total"] == 2
    unchanged = await call(jobs, "idea", action="archive", id=idea["id"])
    assert unchanged == closed
    restored = await call(jobs, "idea", action="restore", id=idea["id"])
    assert "closed_at" not in restored and restored["id"] == idea["id"]
    assert read(tmp_path / "archive/ideas.json") == []
    unlinked = await call(jobs, "todo", action="update", id=task["id"], source_idea=None)
    assert "source_idea" not in unlinked
    response = await jobs["todo"](action="update", id=task["id"], source_idea="missing")
    assert not response.success
    assert await call(jobs, "todo", action="get", id=task["id"]) == unlinked


@pytest.mark.asyncio
async def test_filters_pagination_and_optional_fields(jobs):
    """Listing filters authoritative records and does not impose missing user fields."""
    first = await call(jobs, "todo", action="create", project="ReMe", text="实现接口")
    await call(jobs, "todo", action="create", project="ReMe", text="编写文档")
    other = await call(jobs, "todo", action="create", text="临时事项")
    assert other["project"] == "随笔" and "due_date" not in other
    await call(jobs, "todo", action="update", id=first["id"], status="partial", progress_text="API ready")
    result = await call(jobs, "todo", action="list", project="ReMe", offset=1, limit=1)
    assert result["total"] == 2 and len(result["items"]) == 1
    assert (await call(jobs, "todo", action="list", query="api READY"))["total"] == 1
    assert (await call(jobs, "todo", action="list", status="open"))["total"] == 2
    assert (await call(jobs, "todo", action="list", project=""))["items"][0]["id"] == other["id"]
    assert (await call(jobs, "todo", action="list", offset=100))["items"] == []


@pytest.mark.parametrize("name", ["todo", "idea"])
@pytest.mark.asyncio
async def test_project_defaults_filtering_and_existing_empty_records(jobs, tmp_path, name):
    """Both jobs keep one default project through creation, filtering and partial updates."""
    default = await call(jobs, name, action="create", text="随手记一笔", project=" \t ")
    path = tmp_path / f"{name}s.json"
    assert default["project"] == "随笔" and read(path)[0]["project"] == "随笔"
    custom = await call(jobs, name, action="create", text="项目事项", project=" ReMe ")
    updated = await call(jobs, name, action="update", id=custom["id"], text="更新内容")
    assert updated["project"] == "ReMe"
    reset = await call(jobs, name, action="update", id=custom["id"], project="")
    assert reset["project"] == "随笔"
    assert (await call(jobs, name, action="list", project="随笔"))["total"] == 2
    assert (await call(jobs, name, action="list", project=" \t "))["total"] == 2
    assert (await call(jobs, name, action="list"))["total"] == 2

    # Reading a pre-existing blank project normalizes its value without rewriting user files.
    records = read(path)
    records[0]["project"] = ""
    path.write_text(_store.encode(records), encoding="utf-8")
    before = path.read_bytes()
    assert (await call(jobs, name, action="get", id=records[0]["id"]))["project"] == "随笔"
    assert (await call(jobs, name, action="list", project="随笔"))["total"] == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind,model", [("todos", Todo), ("ideas", Idea)])
@pytest.mark.asyncio
async def test_archive_rollover_retains_history(jobs, tmp_path, kind, model):
    """Both collections retain and resolve records beyond one full archive shard."""
    directory = tmp_path / "archive"
    directory.mkdir()
    now = datetime.now(timezone.utc)
    values = []
    for index in range(1000):
        data = {"id": f"old-{index}", "text": f"历史 {index}", "closed_at": now}
        if kind == "todos":
            data["status"] = "done"
        values.append(model(**data).model_dump(mode="json", exclude_none=True))
    (directory / f"{kind}.json").write_text(_store.encode(values), encoding="utf-8")
    job = "todo" if kind == "todos" else "idea"
    current = await call(jobs, job, action="create", text="新事项")
    fields = {"status": "done"} if kind == "todos" else {}
    await call(jobs, job, action="update" if kind == "todos" else "archive", id=current["id"], **fields)
    assert len(read(directory / f"{kind}.json")) == 1000
    assert read(directory / f"{kind}_0001.json")[0]["id"] == current["id"]
    assert (await call(jobs, job, action="list", scope="archive"))["total"] == 1001
    assert (await call(jobs, job, action="get", id="old-0"))["id"] == "old-0"


@pytest.mark.asyncio
async def test_concurrent_creates_and_progress_preserve_every_record(jobs):
    """Concurrent calls neither lose task creations nor overwrite progress notes."""
    tasks = await asyncio.gather(*(call(jobs, "todo", action="create", text=f"任务 {index}") for index in range(12)))
    identifier = tasks[0]["id"]
    await asyncio.gather(
        *(call(jobs, "todo", action="update", id=identifier, progress_text=f"进展 {index}") for index in range(12)),
    )
    assert (await call(jobs, "todo", action="list"))["total"] == 12
    result = await call(jobs, "todo", action="get", id=identifier)
    assert {note["text"] for note in result["progress"]} == {f"进展 {index}" for index in range(12)}


@pytest.mark.parametrize("failed_file", ["todos.json", "archive/todos.json"])
@pytest.mark.asyncio
async def test_interrupted_archive_recovers_once(jobs, tmp_path, monkeypatch, failed_file):
    """Recovery completes either interrupted publication without duplicating progress."""
    task = await call(jobs, "todo", action="create", text="中断恢复")
    original = _store.write_file_safe

    async def fail(path, *args, **kwargs):
        if path.relative_to(tmp_path).as_posix() == failed_file:
            raise OSError("injected interruption")
        await original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_store, "write_file_safe", fail)
        response = await jobs["todo"](action="update", id=task["id"], status="done", progress_text="已完成")
        assert not response.success
    assert (tmp_path / ".todos.pending.json").exists()
    # Fresh Step and Store instances must recover without any in-memory transaction state.
    recovered = (await TodoStep(app_context=jobs["todo"].app_context)(action="get", id=task["id"])).answer
    assert recovered["status"] == "done" and len(recovered["progress"]) == 1
    assert read(tmp_path / "todos.json") == []
    assert read(tmp_path / "archive/todos.json") == [recovered]
    assert not (tmp_path / ".todos.pending.json").exists()
    assert (await call(jobs, "todo", action="list", scope="all"))["total"] == 1


@pytest.mark.asyncio
async def test_pending_recovery_preserves_external_edits(jobs, tmp_path, monkeypatch):
    """Conflicting manual edits block recovery before any target is overwritten."""
    task = await call(jobs, "todo", action="create", text="用户编辑优先")
    original = _store.write_file_safe

    async def fail(path, *args, **kwargs):
        if path.name == "todos.json":
            raise OSError("injected interruption")
        await original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_store, "write_file_safe", fail)
        assert not (await jobs["todo"](action="update", id=task["id"], status="done")).success
    task["text"] = "用户在服务恢复前手动修订"
    active = tmp_path / "todos.json"
    active.write_text(_store.encode([task]), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    response = await jobs["todo"](action="get", id=task["id"])
    assert not response.success and "changed outside ReMe" in response.answer
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.asyncio
async def test_external_edits_between_read_and_commit_fail_without_overwrite(jobs, tmp_path, monkeypatch):
    """Optimistic checks retain edits made after the job loaded its source files."""
    task = await call(jobs, "todo", action="create", text="原始目标")
    original = _store.RecordStore.put

    async def edit_before_put(store, record):
        task["text"] = "用户手动修改"
        (tmp_path / "todos.json").write_text(_store.encode([task]), encoding="utf-8")
        await original(store, record)

    monkeypatch.setattr(_store.RecordStore, "put", edit_before_put)
    response = await jobs["todo"](action="update", id=task["id"], text="Agent 修改")
    assert not response.success and "changed outside ReMe" in response.answer
    assert read(tmp_path / "todos.json")[0]["text"] == "用户手动修改"
    assert not (tmp_path / ".todos.pending.json").exists()


@pytest.mark.parametrize(
    "payload",
    [
        "[",
        "{}",
        '[{"text":"没有 ID"}]',
        b'[{"id":"bad","text":"\xff"}]',
        '[{"id":"bad","text":"事项","progress":[{"text":"没有时间"}]}]',
    ],
)
@pytest.mark.asyncio
async def test_corrupt_source_is_never_repaired_or_overwritten(jobs, tmp_path, payload):
    """Malformed source content survives failed create calls byte for byte."""
    path = tmp_path / "todos.json"
    original = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    path.write_bytes(original)
    assert not (await jobs["todo"](action="create", text="新条目")).success
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_oversized_source_is_not_truncated_or_overwritten(jobs, tmp_path, monkeypatch):
    """The bounded JSON reader rejects oversized sources before a write-back."""
    payload = _store.encode([Todo(text="保留完整源记录").model_dump(mode="json", exclude_none=True)])
    path = tmp_path / "todos.json"
    path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(_store, "MAX_FILE_READ_BYTES", 32)
    response = await jobs["todo"](action="create", text="不应覆盖")
    assert not response.success and "read limit" in response.answer
    assert path.read_text(encoding="utf-8") == payload


@pytest.mark.asyncio
async def test_oversized_update_never_creates_an_unreadable_recovery_intent(jobs, tmp_path, monkeypatch):
    """An oversized new record fails before publishing any source or journal file."""
    monkeypatch.setattr(_store, "MAX_FILE_READ_BYTES", 512)
    response = await jobs["todo"](action="create", text="x" * 1024)
    assert not response.success and "read limit" in response.answer
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("allowed", [[], ["unrelated"], ["../escape"], ["~/.reme"], {"path": "todos.json"}])
@pytest.mark.asyncio
async def test_invalid_and_restricted_scopes_fail_closed(jobs, tmp_path, allowed):
    """Invalid authorization cannot create collection or recovery files."""
    response = await jobs["todo"](action="create", text="不应保存", _allowed_paths=allowed)
    assert not response.success and not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("relative", ["todos.json", "archive", ".todos.pending.json"])
@pytest.mark.asyncio
async def test_symlink_escapes_do_not_touch_external_files(jobs, tmp_path, tmp_path_factory, relative):
    """Active, archive and journal paths all enforce workspace containment."""
    outside = tmp_path_factory.mktemp("notebook-outside")
    external = outside / "source.json"
    external.write_text("[]", encoding="utf-8")
    (tmp_path / relative).symlink_to(outside if relative == "archive" else external)
    response = await jobs["todo"](action="create", text="不应写入")
    assert not response.success and external.read_text(encoding="utf-8") == "[]"
    assert list(outside.iterdir()) == [external]


@pytest.mark.parametrize(
    "model,payload",
    [
        (TodoRequest, {"action": "create", "id": "caller-id", "text": "事项"}),
        (TodoRequest, {"action": "create"}),
        (TodoRequest, {"action": "list", "progress_text": "无效"}),
        (TodoRequest, {"action": "list", "limit": 101}),
        (TodoRequest, {"action": "get"}),
        (TodoRequest, {"action": "update", "id": "t1"}),
        (TodoRequest, {"action": "update", "id": "t1", "text": None}),
        (TodoRequest, {"action": "update", "id": "t1", "status": "doing"}),
        (TodoRequest, {"action": "update", "id": "t1", "progress_text": None}),
        (TodoRequest, {"action": "update", "id": "t1", "progress_time": "2026-11-18T10:00:00Z"}),
        (
            TodoRequest,
            {"action": "update", "id": "t1", "progress_text": "进展", "progress_time": "2026-11-18T10:00:00"},
        ),
        (IdeaRequest, {"action": "archive"}),
        (IdeaRequest, {"action": "create", "text": " "}),
        (IdeaRequest, {"action": "list", "status": "open"}),
        (IdeaRequest, {"action": "restore", "id": "i1", "text": "不能忽略"}),
    ],
)
def test_action_specific_requests_reject_ignored_or_invalid_fields(model, payload):
    """Each action rejects fields that would otherwise be ignored or misinterpreted."""
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.asyncio
async def test_http_and_mcp_expose_both_jobs_and_preserve_unset_fields(jobs):
    """Actual transport calls preserve omitted fields and support explicit clearing."""
    context = jobs["todo"].app_context
    service = HttpService(app_context=context, web_enabled=False)
    app = SimpleNamespace(config=context.app_config, context=context)
    service.build_service(app)
    service.add_jobs(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.service), base_url="http://test") as client:
        response = await client.post("/idea", json={"action": "create", "text": "HTTP 想法"})
        assert response.status_code == 200 and response.json()["success"]
        idea = response.json()["answer"]
        assert idea["project"] == "随笔"
        response = await client.post(
            "/todo",
            json={"action": "create", "source_idea": idea["id"], "due_date": "2026-11-20"},
        )
        assert response.json()["success"]
        task = response.json()["answer"]
        response = await client.post(
            "/todo",
            json={"action": "update", "id": task["id"], "progress_text": "HTTP 追加进展"},
        )
        assert response.json()["success"]
        saved = response.json()["answer"]
        assert saved["due_date"] == "2026-11-20" and saved["source_idea"] == idea["id"]
        response = await client.post(
            "/todo",
            json={"action": "update", "id": task["id"], "due_date": None, "source_idea": None},
        )
        assert response.json()["success"]
        assert "due_date" not in response.json()["answer"] and "source_idea" not in response.json()["answer"]
    async with Client(service.mcp_server) as client:
        assert {tool.name for tool in await client.list_tools()} == {"todo", "idea"}
        await client.call_tool("todo", {"action": "create", "text": "MCP 任务", "due_date": "2026-11-20"})
        task = (await call(jobs, "todo", action="list"))["items"][0]
        assert task["project"] == "随笔"
        await client.call_tool("todo", {"action": "update", "id": task["id"], "progress_text": "MCP 追加进展"})
        saved = await call(jobs, "todo", action="get", id=task["id"])
        assert saved["due_date"] == "2026-11-20" and len(saved["progress"]) == 1
        await client.call_tool("todo", {"action": "update", "id": task["id"], "due_date": None})
        assert "due_date" not in await call(jobs, "todo", action="get", id=task["id"])


def test_proactive_chat_configuration_resolves_tools_and_skill():
    """Both Chat backends inherit the same tools and discoverable project skills."""
    for config_name in ("proactive", "proactive_cc"):
        config = resolve_app_config(config=config_name, log_config=False)
        assert {"todo", "idea"} <= config["jobs"].keys()
        for name, model in (("todo", TodoRequest), ("idea", IdeaRequest)):
            assert (
                config["jobs"][name]["parameters"]["properties"].keys()
                == model.model_json_schema()["properties"].keys()
            )
        for job in ("ask_agent", "create_chat_session", "process_chat_reminders"):
            assert config["jobs"][job]["steps"][0]["job_tools"] == ["manage_proactive_agent", "todo", "idea"]
        skills = config["components"]["agent_wrapper"]["default"]["skills"]
        assert skills == ["manage-proactive-agent", "manage-todos-ideas"]
        for name in skills:
            assert (Path(__file__).resolve().parents[2] / "skills" / name / "SKILL.md").is_file()


@pytest.mark.parametrize("backend", ["http", "mcp"])
@pytest.mark.asyncio
async def test_native_clients_send_action_as_job_payload(jobs, backend):
    """CLI client dispatch works with the same action-bearing jobs as direct tools."""
    context = jobs["todo"].app_context
    service = HttpService(app_context=context, web_enabled=False)
    app = SimpleNamespace(config=context.app_config, context=context)
    service.build_service(app)
    service.add_jobs(app)
    if backend == "http":
        client = HttpClient()
        client.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=service.service), base_url="http://test")
    else:
        client = MCPClient(transport=service.mcp_server)
    async with client:
        chunks = [chunk async for chunk in client.call_job("todo", {"action": "create", "text": "CLI 事项"})]
    assert chunks
    task = (await call(jobs, "todo", action="list"))["items"][0]
    assert task["text"] == "CLI 事项" and task["project"] == "随笔"
