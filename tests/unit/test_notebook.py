"""Todo/Idea job contracts, authoritative JSON and interrupted archive recovery."""

# pylint: disable=protected-access,redefined-outer-name

import asyncio
from datetime import datetime, timezone
import json
from types import SimpleNamespace

from fastmcp import Client
import httpx
from pydantic import ValidationError
import pytest
import pytest_asyncio

from reme.core.context import ApplicationContext
from reme.components.client.http_client import HttpClient
from reme.components.client.mcp_client import MCPClient
from reme.components.job import BaseJob
from reme.components.service import HttpService
from reme.config import resolve_app_config
from reme.schema import Idea, IdeaRequest, Todo, TodoRequest
from reme.steps.notebook import TodoStep
from reme.steps.notebook import _store


@pytest_asyncio.fixture
async def jobs(tmp_path, request):
    """Run real registered steps with no models, network or repository workspace."""
    config = resolve_app_config(config="proactive", log_config=False)
    context = ApplicationContext(workspace_dir=str(tmp_path), **getattr(request, "param", {}))
    result = {name: BaseJob(name=name, app_context=context, **config["jobs"][name]) for name in ("todo", "idea")}
    context.jobs = result
    for job in result.values():
        await job.start()
    yield result
    for job in reversed(list(result.values())):
        await job.close()


@pytest.fixture
def notebook_path(tmp_path):
    """Expected default location for the notebook files."""
    return tmp_path / "notebook"


async def call(jobs, name, **kwargs):
    """Require successful business responses and return their JSON-ready payloads."""
    response = await jobs[name](**kwargs)
    assert response.success, response.answer
    return response.answer


def read(path):
    """Inspect durable arrays independently of the management jobs."""
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_todo_progress_dates_archive_and_reopen(jobs, notebook_path):
    """State transitions retain identity, deadlines and the complete progress history."""
    idea = await call(jobs, "idea", action="create", project="ReMe", text="通过对话管理待办")
    task = await call(jobs, "todo", action="create", source_idea=idea["id"], due_date="2026-11-20")
    identifier = task["id"]
    assert task["project"] == "ReMe" and task["text"] == idea["text"]
    assert task["status"] == "pending" and task["progress"] == []
    await call(jobs, "todo", action="update", id=identifier, status="in_progress", progress_text="接口完成，测试待补")
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
    closed = await call(jobs, "todo", action="update", id=identifier, status="closed", close_note="已上线，测试通过")
    assert closed["closed_at"] and closed["close_note"] == "已上线，测试通过"
    assert closed["progress"] == task["progress"] and read(notebook_path / "todos.json") == []
    assert read(notebook_path / "archive/todos.json") == [closed]
    assert (await call(jobs, "todo", action="list"))["total"] == 0
    archived = await call(jobs, "todo", action="list", scope="archive", status="closed", query="测试通过")
    assert archived["total"] == 1 and archived["items"][0]["close_note"] == closed["close_note"]
    assert await call(jobs, "todo", action="get", id=identifier) == closed
    assert "closed_at" not in await call(jobs, "idea", action="get", id=idea["id"])
    reopened = await call(jobs, "todo", action="update", id=identifier, status="pending", due_date=None)
    assert not {"closed_at", "close_note", "due_date"} & reopened.keys()
    assert reopened["progress"] == closed["progress"] and reopened["id"] == identifier
    assert read(notebook_path / "archive/todos.json") == []
    closed_again = await call(jobs, "todo", action="update", id=identifier, status="closed")
    assert closed_again["closed_at"] and closed_again["status"] == "closed" and "close_note" not in closed_again


@pytest.mark.asyncio
async def test_existing_links_idea_closure_and_restore(jobs, notebook_path):
    """Links survive independent idea closure without changing existing task content."""
    idea = await call(jobs, "idea", action="create", project="ReMe", text="想法")
    task = await call(jobs, "todo", action="create", project="Other", text="已有任务")
    linked = await call(jobs, "todo", action="update", id=task["id"], source_idea=idea["id"])
    assert linked["project"] == "Other" and linked["text"] == "已有任务" and linked["status"] == "pending"
    await call(jobs, "idea", action="update", id=idea["id"], text="补充想法")
    closed = await call(jobs, "idea", action="archive", id=idea["id"])
    assert read(notebook_path / "ideas.json") == []
    assert read(notebook_path / "archive/ideas.json") == [closed]
    inherited = await call(jobs, "todo", action="create", source_idea=idea["id"])
    assert inherited["text"] == "补充想法"
    assert (await call(jobs, "todo", action="list", source_idea=idea["id"]))["total"] == 2
    unchanged = await call(jobs, "idea", action="archive", id=idea["id"])
    assert unchanged == closed
    restored = await call(jobs, "idea", action="restore", id=idea["id"])
    assert "closed_at" not in restored and restored["id"] == idea["id"]
    assert read(notebook_path / "archive/ideas.json") == []
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
    await call(jobs, "todo", action="update", id=first["id"], status="in_progress", progress_text="API ready")
    result = await call(jobs, "todo", action="list", project="ReMe", offset=1, limit=1)
    assert result["total"] == 2 and len(result["items"]) == 1
    assert (await call(jobs, "todo", action="list", query="api READY"))["total"] == 1
    assert (await call(jobs, "todo", action="list", status="pending"))["total"] == 2
    assert (await call(jobs, "todo", action="list", project=""))["items"][0]["id"] == other["id"]
    assert (await call(jobs, "todo", action="list", offset=100))["items"] == []


@pytest.mark.asyncio
async def test_notebook_timeline_order_before_pagination(jobs):
    """Agenda pages follow due dates and archive edits preserve the closure order."""
    first = await call(jobs, "todo", action="create", text="First", due_date="2026-11-25")
    later = await call(jobs, "todo", action="create", text="Later", due_date="2026-12-04")
    undated = await call(jobs, "todo", action="create", text="Undated")
    for offset, task in enumerate((first, later, undated)):
        page = await call(jobs, "todo", action="list", offset=offset, limit=1)
        assert page["items"][0]["id"] == task["id"]

    for kind, records in (("todo", (first, later)), ("idea", None)):
        if records is None:
            records = [await call(jobs, kind, action="create", text=text) for text in ("First", "Later")]
        older, newer = (record["id"] for record in records)
        for identifier in (older, newer):
            if kind == "todo":
                await call(jobs, kind, action="update", id=identifier, status="closed")
            else:
                await call(jobs, kind, action="archive", id=identifier)
        await call(jobs, kind, action="update", id=older, text="First, edited")
        archive = await call(jobs, kind, action="list", scope="archive", limit=1)
        assert archive["items"][0]["id"] == newer


@pytest.mark.asyncio
async def test_idea_capture_time_and_undated_order(jobs, notebook_path):
    """Capture time survives edits and restoration; undated ideas retain their order."""
    notebook_path.mkdir()
    path = notebook_path / "ideas.json"
    path.write_text(
        json.dumps(
            [
                {"id": "undated-first", "text": "First undated idea"},
                {"id": "undated-last", "text": "Last undated idea"},
            ],
        ),
        encoding="utf-8",
    )
    await call(jobs, "idea", action="update", id="undated-first", text="Edited undated idea")
    first = await call(jobs, "idea", action="create", text="First captured idea")
    later = await call(jobs, "idea", action="create", text="Later captured idea")
    assert datetime.fromisoformat(first["created_at"]).utcoffset() is not None
    for action in ("update", "archive", "restore"):
        changes = {"text": "Edited captured idea"} if action == "update" else {}
        record = await call(jobs, "idea", action=action, id=first["id"], **changes)
        assert record["created_at"] == first["created_at"]
    for offset, identifier in enumerate((later["id"], first["id"], "undated-last", "undated-first")):
        page = await call(jobs, "idea", action="list", offset=offset, limit=1)
        assert page["items"][0]["id"] == identifier
    assert all("created_at" not in record for record in read(path) if record["id"].startswith("undated-"))


@pytest.mark.asyncio
async def test_list_projects_counts_both_collections_by_scope(jobs, notebook_path):
    """Project discovery includes Idea-only and archive-only projects and respects scope."""
    assert await call(jobs, "todo", action="list_projects") == {"items": [], "total": 0}
    assert not notebook_path.exists()
    await call(jobs, "todo", action="create", project="ReMe", text="待办")
    await call(jobs, "todo", action="create", project="Archive", text="已完成待办", status="closed")
    await call(jobs, "todo", action="create", text="随手记下的待办")
    await call(jobs, "idea", action="create", project="Ideas", text="想法")
    idea = await call(jobs, "idea", action="create", project="ReMe", text="待办的来源")
    await call(jobs, "idea", action="archive", id=idea["id"])

    archive_project = {"project": "Archive", "todo_count": 1, "idea_count": 0}
    idea_project = {"project": "Ideas", "todo_count": 0, "idea_count": 1}
    notes_project = {"project": "随笔", "todo_count": 1, "idea_count": 0}
    projects = {
        "items": [archive_project, idea_project, {"project": "ReMe", "todo_count": 1, "idea_count": 1}, notes_project],
        "total": 4,
    }
    assert await call(jobs, "todo", action="list_projects") == projects
    assert await call(jobs, "todo", action="list_projects", scope="all") == projects
    assert await call(jobs, "todo", action="list_projects", scope="active") == {
        "items": [idea_project, {"project": "ReMe", "todo_count": 1, "idea_count": 0}, notes_project],
        "total": 3,
    }
    assert await call(jobs, "todo", action="list_projects", scope="archive") == {
        "items": [archive_project, {"project": "ReMe", "todo_count": 0, "idea_count": 1}],
        "total": 2,
    }


@pytest.mark.parametrize("name", ["todo", "idea"])
@pytest.mark.asyncio
async def test_project_defaults_filtering_and_existing_empty_records(jobs, notebook_path, name):
    """Both jobs keep one default project through creation, filtering and partial updates."""
    default = await call(jobs, name, action="create", text="随手记一笔", project=" \t ")
    path = notebook_path / f"{name}s.json"
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
async def test_archive_rollover_retains_history(jobs, notebook_path, kind, model):
    """Both collections retain and resolve records beyond one full archive shard."""
    directory = notebook_path / "archive"
    directory.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    values = []
    for index in range(1000):
        data = {"id": f"old-{index}", "text": f"历史 {index}", "closed_at": now}
        if kind == "todos":
            data["status"] = "closed"
        values.append(model(**data).model_dump(mode="json", exclude_none=True))
    (directory / f"{kind}.json").write_text(_store.encode(values), encoding="utf-8")
    job = "todo" if kind == "todos" else "idea"
    current = await call(jobs, job, action="create", text="新事项")
    fields = {"status": "closed"} if kind == "todos" else {}
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
async def test_interrupted_archive_recovers_once(jobs, notebook_path, monkeypatch, failed_file):
    """Recovery completes either interrupted publication without duplicating progress."""
    task = await call(jobs, "todo", action="create", text="中断恢复")
    original = _store.write_file_safe

    async def fail(path, *args, **kwargs):
        if path.relative_to(notebook_path).as_posix() == failed_file:
            raise OSError("injected interruption")
        await original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_store, "write_file_safe", fail)
        response = await jobs["todo"](
            action="update",
            id=task["id"],
            status="closed",
            progress_text="已部署",
            close_note="验收通过",
        )
        assert not response.success
    assert (notebook_path / ".todos.pending.json").exists()
    # Fresh Step and Store instances must recover without any in-memory transaction state.
    recovered = (await TodoStep(app_context=jobs["todo"].app_context)(action="get", id=task["id"])).answer
    assert recovered["status"] == "closed" and recovered["close_note"] == "验收通过" and len(recovered["progress"]) == 1
    assert read(notebook_path / "todos.json") == []
    assert read(notebook_path / "archive/todos.json") == [recovered]
    assert not (notebook_path / ".todos.pending.json").exists()
    assert (await call(jobs, "todo", action="list", scope="all"))["total"] == 1


@pytest.mark.asyncio
async def test_pending_recovery_preserves_external_edits(jobs, notebook_path, monkeypatch):
    """Conflicting manual edits block recovery before any target is overwritten."""
    task = await call(jobs, "todo", action="create", text="用户编辑优先")
    original = _store.write_file_safe

    async def fail(path, *args, **kwargs):
        if path.name == "todos.json":
            raise OSError("injected interruption")
        await original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_store, "write_file_safe", fail)
        assert not (await jobs["todo"](action="update", id=task["id"], status="closed")).success
    task["text"] = "用户在服务恢复前手动修订"
    active = notebook_path / "todos.json"
    active.write_text(_store.encode([task]), encoding="utf-8")
    before = {path: path.read_bytes() for path in notebook_path.rglob("*.json")}
    response = await jobs["todo"](action="get", id=task["id"])
    assert not response.success and "changed outside ReMe" in response.answer
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.asyncio
async def test_external_edits_between_read_and_commit_fail_without_overwrite(jobs, notebook_path, monkeypatch):
    """Optimistic checks retain edits made after the job loaded its source files."""
    task = await call(jobs, "todo", action="create", text="原始目标")
    original = _store.RecordStore.put

    async def edit_before_put(store, record):
        task["text"] = "用户手动修改"
        (notebook_path / "todos.json").write_text(_store.encode([task]), encoding="utf-8")
        await original(store, record)

    monkeypatch.setattr(_store.RecordStore, "put", edit_before_put)
    response = await jobs["todo"](action="update", id=task["id"], text="Agent 修改")
    assert not response.success and "changed outside ReMe" in response.answer
    assert read(notebook_path / "todos.json")[0]["text"] == "用户手动修改"
    assert not (notebook_path / ".todos.pending.json").exists()


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
async def test_corrupt_source_is_never_repaired_or_overwritten(jobs, notebook_path, payload):
    """Malformed source content survives failed create calls byte for byte."""
    path = notebook_path / "todos.json"
    path.parent.mkdir()
    original = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    path.write_bytes(original)
    assert not (await jobs["todo"](action="create", text="新条目")).success
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_oversized_source_is_not_truncated_or_overwritten(jobs, notebook_path, monkeypatch):
    """The bounded JSON reader rejects oversized sources before a write-back."""
    payload = _store.encode([Todo(text="保留完整源记录").model_dump(mode="json", exclude_none=True)])
    path = notebook_path / "todos.json"
    path.parent.mkdir()
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


@pytest.mark.parametrize("allowed", [[], ["todos.json"], ["../escape"], ["~/.reme"], {"path": "notebook"}])
@pytest.mark.asyncio
async def test_invalid_and_restricted_scopes_fail_closed(jobs, tmp_path, allowed):
    """Invalid authorization cannot create collection or recovery files."""
    response = await jobs["todo"](action="create", text="不应保存", _allowed_paths=allowed)
    assert not response.success and not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    "relative",
    ["notebook", "notebook/todos.json", "notebook/archive", "notebook/.todos.pending.json"],
)
@pytest.mark.asyncio
async def test_symlink_escapes_do_not_touch_external_files(jobs, tmp_path, tmp_path_factory, relative):
    """Active, archive and journal paths all enforce workspace containment."""
    outside = tmp_path_factory.mktemp("notebook-outside")
    external = outside / "source.json"
    external.write_text("[]", encoding="utf-8")
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(outside if path.name in {"notebook", "archive"} else external)
    response = await jobs["todo"](action="create", text="不应写入")
    assert not response.success and external.read_text(encoding="utf-8") == "[]"
    assert list(outside.iterdir()) == [external]


@pytest.mark.parametrize("jobs", [{"notebook_dir": "planning/capture"}], indirect=True)
@pytest.mark.asyncio
async def test_custom_notebook_directory_contains_linked_records_and_archives(jobs, tmp_path):
    """A custom directory covers both jobs, their archives and workspace-relative scopes."""
    allowed = {"_allowed_paths": ["planning/capture"]}
    idea = await call(jobs, "idea", action="create", text="想法", **allowed)
    await call(jobs, "idea", action="archive", id=idea["id"], **allowed)
    task = await call(jobs, "todo", action="create", source_idea=idea["id"], **allowed)
    await call(jobs, "todo", action="update", id=task["id"], status="closed", **allowed)

    directory = tmp_path / "planning/capture"
    assert read(directory / "todos.json") == read(directory / "ideas.json") == []
    assert read(directory / "archive/todos.json")[0]["source_idea"] == idea["id"]
    assert read(directory / "archive/ideas.json")[0]["id"] == idea["id"]
    assert await call(jobs, "todo", action="list_projects", **allowed) == {
        "items": [{"project": "随笔", "todo_count": 1, "idea_count": 1}],
        "total": 1,
    }
    assert set(tmp_path.iterdir()) == {tmp_path / "planning"}


@pytest.mark.parametrize(
    "model,payload",
    [
        (TodoRequest, {"action": "create", "id": "caller-id", "text": "事项"}),
        (TodoRequest, {"action": "create"}),
        (TodoRequest, {"action": "list", "progress_text": "无效"}),
        (TodoRequest, {"action": "list", "limit": 101}),
        (TodoRequest, {"action": "list_projects", "project": "ReMe"}),
        (TodoRequest, {"action": "get"}),
        (TodoRequest, {"action": "update", "id": "t1"}),
        (TodoRequest, {"action": "update", "id": "t1", "text": None}),
        (TodoRequest, {"action": "update", "id": "t1", "status": "doing"}),
        (TodoRequest, {"action": "update", "id": "t1", "status": "done"}),
        (TodoRequest, {"action": "update", "id": "t1", "status": "pending", "close_note": "已完成"}),
        (TodoRequest, {"action": "update", "id": "t1", "status": "closed", "close_note": " "}),
        (TodoRequest, {"action": "update", "id": "t1", "progress_text": None}),
        (TodoRequest, {"action": "update", "id": "t1", "progress_time": "2026-11-18T10:00:00Z"}),
        (
            TodoRequest,
            {"action": "update", "id": "t1", "progress_text": "进展", "progress_time": "2026-11-18T10:00:00"},
        ),
        (IdeaRequest, {"action": "archive"}),
        (IdeaRequest, {"action": "create", "text": " "}),
        (IdeaRequest, {"action": "list", "status": "pending"}),
        (IdeaRequest, {"action": "list_projects"}),
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
        await call(jobs, "idea", action="archive", id=idea["id"])
        projects = {"items": [{"project": "随笔", "todo_count": 1, "idea_count": 1}], "total": 1}
        response = await client.post("/todo", json={"action": "list_projects"})
        assert response.json()["success"] and response.json()["answer"] == projects
    async with Client(service.mcp_server) as client:
        assert {tool.name for tool in await client.list_tools()} == {"todo", "idea"}
        result = await client.call_tool("todo", {"action": "list_projects"})
        assert result.structured_content == projects
        await client.call_tool("todo", {"action": "create", "text": "MCP 任务", "due_date": "2026-11-20"})
        task = (await call(jobs, "todo", action="list"))["items"][0]
        assert task["project"] == "随笔"
        await client.call_tool("todo", {"action": "update", "id": task["id"], "progress_text": "MCP 追加进展"})
        saved = await call(jobs, "todo", action="get", id=task["id"])
        assert saved["due_date"] == "2026-11-20" and len(saved["progress"]) == 1
        await client.call_tool(
            "todo",
            {"action": "update", "id": task["id"], "status": "closed", "close_note": "验收通过", "due_date": None},
        )
        closed = await call(jobs, "todo", action="get", id=task["id"])
        assert closed["status"] == "closed" and closed["close_note"] == "验收通过" and "due_date" not in closed
        await client.call_tool("todo", {"action": "update", "id": task["id"], "status": "closed", "close_note": None})
        cleared = await call(jobs, "todo", action="get", id=task["id"])
        assert cleared["closed_at"] == closed["closed_at"] and "close_note" not in cleared


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
