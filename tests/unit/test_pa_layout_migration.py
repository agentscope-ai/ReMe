"""Persistent PA layout migration preserves documents, identities and recovery paths."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from datetime import datetime
import json

import frontmatter
import pytest

from reme.components import ApplicationContext, PaRuntime
from reme.schema.pa_task import PaTask, PaTaskRecovery, PaTaskRun
from reme.schema.pa_temp_task import PaTempTask
from scripts.migrate_pa_tasks import _Migration, migrate

from .pa_helpers import PA_CONFIG


@pytest.fixture
def legacy(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path), metadata_dir="state", daily_dir="notes/daily")
    runtime = PaRuntime(app_context=context, default_proactive=PA_CONFIG)
    root = runtime.component_metadata_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.json").write_text('[{"id":"one","cron":"*/5 * * * *","query":"work","enabled":true}]')
    (root / "proactive_inbox.txt").write_text("请关注机器人。", encoding="utf-8")
    (root / "proactive_requests.jsonl").write_text('{"id":"request-one","status":"queued"}\n')
    now = datetime.fromisoformat("2026-11-08T12:00:00+08:00")
    run = PaTaskRun(task_id="one", query="work", started_at=now, agent_session_id="cron-session", phase="summary")
    result = "reme_cron/one/2026-11-08/120000.md"
    saved = PaTaskRecovery(
        task=PaTask(id="one", cron="*/5 * * * *", query="work", enabled=True),
        run=run,
        target_path=result,
    )
    files = {
        result: frontmatter.dumps(
            frontmatter.Post("发现。", task_id="one", run_id=run.run_id, status="unread", user_read_at=None),
        ),
        "reme_cron/one/recovery.json": saved.model_dump_json(),
        "reme_cron/one/handoff.md": "保留交接。",
        "reme_cron/temporary/oneshot/handoff.md": "临时任务背景。",
        "reme_cron/temporary/oneshot/result.md": "临时结果。",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    temporary = PaTempTask(
        id="oneshot",
        query="work",
        source_session_id="main-session",
        run_at=now,
        created_at=now,
        updated_at=now,
        handoff_path="reme_cron/temporary/oneshot/handoff.md",
        result_path="reme_cron/temporary/oneshot/result.md",
        status="sending",
    )
    runtime.temp_tasks.path.write_text(json.dumps([temporary.model_dump(mode="json")]))
    (root / "task_runs.jsonl").write_text(run.model_copy(update={"result_path": result}).model_dump_json() + "\n")
    (root / "reminders.jsonl").write_text(
        json.dumps(
            {
                "id": "message-one",
                "content": "消息正文。",
                "kind": "normal",
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "expires_at": "2026-11-08T12:05:00+08:00",
                "session_id": "main-session",
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    runtime.proactive_context_path.write_text(
        json.dumps(
            {
                "phase": "task_results",
                "documents": [{"path": result, "content": "原文 reme_cron 不修改。"}],
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    return runtime


def contents(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_layout_migration_preserves_sources_and_reloads_recovery_and_temporary_tasks(legacy):
    runtime = legacy
    workspace = runtime.workspace_path
    original = contents(workspace)
    assert migrate(workspace, metadata_dir="state", daily_dir="notes/daily", check=True) > 0
    assert contents(workspace) == original
    assert migrate(workspace, metadata_dir="state", daily_dir="notes/daily") > 0
    assert runtime.cron_path.exists() and runtime.chat_inbox_path.exists() and runtime.inbox.receipts_path.exists()
    for path, text in original.items():
        if path.name not in {"temp_tasks.json", "proactive_context.jsonl"}:
            assert (workspace / path).read_bytes() == text
    assert list(runtime.temp_tasks.path.parent.glob("temp_tasks.json.before-pa-layout-*.bak"))
    task = runtime.tasks.load()[0]
    recovered = runtime.tasks.recovery(task)
    assert recovered.target_path == "state/pa/cron/one/2026-11-08/120000.md"
    temporary = runtime.temp_tasks.load()[0]
    assert temporary.source_session_id == "main-session" and temporary.revision == 0
    assert temporary.result_path == "state/pa/temporary/oneshot/result.md"
    assert runtime.temp_tasks.document_path(temporary.id, "result.md").read_text() == "临时结果。"
    record = json.loads(runtime.cron_runs_path.read_text())
    assert record["result_path"] == recovered.target_path
    checkpoint = json.loads(runtime.proactive_context_path.read_text())
    assert checkpoint["documents"][0]["path"] == recovered.target_path
    assert checkpoint["documents"][0]["content"] == "原文 reme_cron 不修改。"
    runtime._initialize_files()
    assert runtime.reminders.read()[0].content == "消息正文。"
    assert migrate(workspace, metadata_dir="state", daily_dir="notes/daily") == 0


def test_old_layout_is_rejected_before_initialization_can_seed_empty_new_state(legacy):
    before = contents(legacy.workspace_path)
    with pytest.raises(ValueError, match="offline migration"):
        legacy._initialize_files()
    assert contents(legacy.workspace_path) == before


@pytest.mark.parametrize(
    "filename",
    ["tasks.json", "proactive_inbox.txt", "reminders.jsonl", "task_runs.jsonl", "proactive_requests.jsonl"],
)
def test_each_unmigrated_state_file_blocks_empty_replacement(tmp_path, filename):
    runtime = PaRuntime(app_context=ApplicationContext(workspace_dir=str(tmp_path)), default_proactive=PA_CONFIG)
    path = runtime.component_metadata_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]" if filename == "tasks.json" else "")
    before = contents(tmp_path)
    with pytest.raises(ValueError, match="offline migration"):
        runtime._initialize_files()
    assert contents(tmp_path) == before


def test_partial_migration_cannot_start_before_the_canonical_chat_inbox_is_published(tmp_path):
    runtime = PaRuntime(app_context=ApplicationContext(workspace_dir=str(tmp_path)), default_proactive=PA_CONFIG)
    runtime.cron_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.cron_path.write_text("[]")
    runtime.cron_path.with_name("tasks.json").write_text("[]")
    before = contents(tmp_path)
    with pytest.raises(ValueError, match="offline migration"):
        runtime._initialize_files()
    assert contents(tmp_path) == before


@pytest.mark.parametrize("conflict", ["document", "journal"])
def test_conflicts_preserve_all_sources_and_destinations(legacy, conflict):
    target = legacy.tasks.task_path("one") / "handoff.md" if conflict == "document" else legacy.chat_inbox_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("用户的新内容" if conflict == "document" else '{"id":"other"}\n')
    before = contents(legacy.workspace_path)
    with pytest.raises(FileExistsError):
        migrate(legacy.workspace_path, metadata_dir="state", daily_dir="notes/daily")
    assert contents(legacy.workspace_path) == before


def test_invalid_later_input_prevents_partial_publication(legacy):
    legacy.temp_tasks.path.write_text("[")
    before = contents(legacy.workspace_path)
    with pytest.raises(ValueError):
        migrate(legacy.workspace_path, metadata_dir="state", daily_dir="notes/daily")
    assert contents(legacy.workspace_path) == before


def test_source_change_is_detected_before_publication(legacy):
    migration = _Migration(legacy)
    source = legacy.component_metadata_path / "reminders.jsonl"
    migration.plan(source, legacy.chat_inbox_path)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(RuntimeError, match="sources changed"):
        migration.publish(False)
    assert not legacy.chat_inbox_path.exists()


def test_symlink_source_cannot_escape_workspace(legacy, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("outside")
    link = legacy.workspace_path / "reme_cron/one/escape.md"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="without symlinks"):
        migrate(legacy.workspace_path, metadata_dir="state", daily_dir="notes/daily")
    assert not legacy.chat_inbox_path.exists()


def test_older_daily_results_and_absolute_references_are_migrated(legacy):
    workspace = legacy.workspace_path
    source = workspace / "notes/daily/2026-11-07/cron/two/235900.md"
    source.parent.mkdir(parents=True)
    source.write_text(frontmatter.dumps(frontmatter.Post("旧日结果。", task_id="two", run_id="previous-run")))
    legacy.proactive_context_path.write_text(json.dumps({"path": str(source)}) + "\n")
    migrate(workspace, metadata_dir="state", daily_dir="notes/daily")
    target = legacy.tasks.task_path("two") / "2026-11-07/235900.md"
    assert target.read_bytes() == source.read_bytes()
    assert json.loads(legacy.proactive_context_path.read_text())["path"] == str(target)


def test_legacy_inline_run_results_become_archived_documents_without_new_proactive_inputs(legacy):
    source = legacy.component_metadata_path / "task_runs.jsonl"
    record = json.loads(source.read_text())
    record.pop("result_path")
    record.update(cron="*/5 * * * *", scheduled_at=record["started_at"], result="完整历史结果。", status="succeeded")
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    original = source.read_bytes()
    migrate(legacy.workspace_path, metadata_dir="state", daily_dir="notes/daily")
    run = PaTaskRun.model_validate_json(legacy.cron_runs_path.read_text())
    post = frontmatter.load(legacy.workspace_path / run.result_path)
    assert post.content == "完整历史结果。" and post["run_id"] == record["run_id"]
    assert post["status"] == "read" and post["user_read_at"] is None
    assert source.read_bytes() == original
    assert migrate(legacy.workspace_path, metadata_dir="state", daily_dir="notes/daily") == 0
