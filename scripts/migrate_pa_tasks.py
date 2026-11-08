"""Offline PA layout migration; preserve originals and publish chat_inbox.jsonl last."""

# pylint: disable=protected-access

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
from uuid import uuid4

import frontmatter

from reme.components import ApplicationContext, PaRuntime
from reme.constants import DEFAULT_MAX_FILE_BYTES
from reme.schema.pa_hint import PaHintState
from reme.schema.pa_task import PaTaskRecovery, PaTaskRun
from reme.schema.pa_temp_task import PaTempTask
from reme.schema.reminder import ReminderRecord


class _Migration:
    """Preflight a stopped workspace before publishing any changed file."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.files = runtime.tasks
        self.sources = {}
        self.plans = {}
        self.renamed = {
            "reminders.jsonl": runtime.chat_inbox_path,
            "task_runs.jsonl": runtime.cron_runs_path,
            "proactive_requests.jsonl": runtime.inbox.receipts_path,
        }

    def read(self, path):
        """Retain exact source bytes and reject aliases or non-file sources."""
        path = self.files.checked(path)
        if not path.is_file():
            raise ValueError(f"Migration source must be a regular file: {path}")
        content = path.read_bytes()
        content.decode("utf-8")
        previous = self.sources.setdefault(path, content)
        if previous != content:
            raise RuntimeError("PA sources changed during migration; stop ReMe first")
        return content

    def relocate(self, value):
        """Translate complete workspace paths, retaining relative/absolute form."""
        path = Path(value)
        absolute = path.is_absolute()
        if absolute:
            try:
                path = path.relative_to(self.runtime.workspace_path)
            except ValueError:
                return value
        parts = path.parts
        daily = Path(self.runtime.app_context.app_config.daily_dir).parts
        tail = parts[len(daily) :] if parts[: len(daily)] == daily else ()
        if parts and parts[0] == "reme_cron":
            temporary = parts[1:2] == ("temporary",)
            target = self.runtime._metadata_file("temporary" if temporary else "cron")
            target = target.joinpath(*(parts[2:] if temporary else parts[1:]))
        elif len(tail) >= 4 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tail[0]) and tail[1] == "cron":
            target = self.files.task_path(tail[2]) / tail[0] / Path(*tail[3:])
        else:
            for old, new in self.renamed.items():
                if path == self.runtime._metadata_file(old).relative_to(self.runtime.workspace_path):
                    target = new
                    break
            else:
                return value
        self.files.checked(target)
        return str(target) if absolute else target.relative_to(self.runtime.workspace_path).as_posix()

    def rewrite(self, value, key=""):
        """Rewrite structured file references without changing user text or queries."""
        if isinstance(value, dict):
            return {name: self.rewrite(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [self.rewrite(item, key) for item in value]
        if isinstance(value, str) and (key in {"path", "paths"} or key.endswith(("_path", "_paths"))):
            return self.relocate(value)
        return value

    def converted(self, path, data):
        """Validate structured records and update their stored file references."""
        if path.suffix not in {".json", ".jsonl"}:
            return data
        values = (
            [json.loads(line) for line in data.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(data)
        )
        updated = self.rewrite(values)
        name = path.name
        if path.suffix == ".jsonl" and any(not isinstance(row, dict) for row in updated):
            raise ValueError(f"Expected JSON objects in {path}")
        if name in {"reminders.jsonl", "chat_inbox.jsonl"}:
            hint_path = self.runtime._metadata_file("hint_state.json")
            hints = PaHintState.model_validate_json(self.read(hint_path)) if hint_path.exists() else PaHintState()
            periods = {draw.round_id: period for period, draw in hints.draws.items() if draw.hint_index is not None}
            for row in updated:
                row.setdefault("kind", periods.get(row.get("id"), "normal"))
                ReminderRecord.model_validate(row)
        elif name in {"task_runs.jsonl", "cron_runs.jsonl"}:
            for row in updated:
                if name == "task_runs.jsonl":
                    inline = row.pop("result", None)
                    row.pop("cron", None)
                    row.pop("scheduled_at", None)
                    if inline:
                        run = PaTaskRun.model_validate(row)
                        timestamp = run.finished_at or run.started_at
                        suffix = sha256(run.run_id.encode()).hexdigest()[:32]
                        target = (
                            self.files.task_path(run.task_id) / f"{timestamp:%Y-%m-%d}/{timestamp:%H%M%S}-{suffix}.md"
                        )
                        post = frontmatter.Post(
                            inline,
                            task_id=run.task_id,
                            run_id=run.run_id,
                            created_at=timestamp.isoformat(),
                            status="read",
                            user_read_at=None,
                        )
                        self.stage(None, target, (frontmatter.dumps(post) + "\n").encode())
                        row["result_path"] = target.relative_to(self.runtime.workspace_path).as_posix()
                PaTaskRun.model_validate(row)
        elif name == "recovery.json":
            PaTaskRecovery.model_validate(updated)
        elif name == "temp_tasks.json":
            if not isinstance(updated, list):
                raise ValueError("temp_tasks.json must contain a list")
            records = [PaTempTask.model_validate(row) for row in updated]
            if len({record.id for record in records}) != len(records):
                raise ValueError("Duplicate temporary task IDs")
            for record in records:
                for field, document in (("handoff_path", "handoff.md"), ("result_path", "result.md")):
                    reference = getattr(record, field)
                    if (
                        reference is not None
                        and self.runtime.workspace_path / reference
                        != self.runtime.temp_tasks.document_path(record.id, document)
                    ):
                        raise ValueError("Temporary document path does not match its task")
        elif name in {"proactive_requests.jsonl", "proactive_inbox_receipts.jsonl"}:
            for row in updated:
                if not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("status"), str):
                    raise ValueError("Inbox receipts require an identity and status")
                if row["status"] == "staging" and any(
                    not isinstance(row.get(key), str) for key in ("message", "created_at")
                ):
                    raise ValueError("Staged inbox receipts require their message and creation time")
        if updated == values:
            return data
        if path.suffix == ".jsonl":
            return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated).encode()
        return (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode()

    def plan(self, source, target):
        """Keep authoritative destinations and reject conflicting copies."""
        content = self.converted(source, self.read(source))
        self.stage(source, target, content)

    def stage(self, source, target, content):
        """Preflight a copied or extracted document against its destination."""
        target = self.files.checked(target)
        if target.suffix in {".json", ".md", ".txt"} and len(content) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError(f"Migrated document exceeds the file read limit: {target}")
        previous = self.plans.get(target)
        if previous is not None:
            if previous != content:
                raise FileExistsError(f"Conflicting migration sources: {target}")
            return
        if target.exists():
            current = self.read(target)
            if current == content:
                return
            if source != target:
                if target.suffix == ".md":
                    old, new = frontmatter.loads(content.decode()), frontmatter.loads(current.decode())
                    same_run = old.get("run_id") and old.get("run_id") == new.get("run_id")
                    if old.get("task_id") == new.get("task_id") and (same_run or old.content == new.content):
                        return
                elif target.suffix == ".jsonl":
                    old = [json.loads(line) for line in content.splitlines() if line.strip()]
                    new = [json.loads(line) for line in current.splitlines() if line.strip()]
                    if new[: len(old)] == old:
                        return
                raise FileExistsError(f"Conflicting PA destination: {target}")
        self.plans[target] = content

    def verify(self):
        """Fail if any observed source or pre-existing destination changed."""
        if any(self.files.checked(path).read_bytes() != data for path, data in self.sources.items()):
            raise RuntimeError("PA sources changed during migration; stop ReMe first")

    def publish(self, check):
        """Back up in-place edits; publish the canonical inbox after all data."""
        self.verify()
        if check:
            return len(self.plans)
        self.files.root_path.mkdir(parents=True, exist_ok=True)
        self.runtime.temp_tasks.root_path.mkdir(parents=True, exist_ok=True)
        for target in sorted(self.plans, key=lambda path: path == self.runtime.chat_inbox_path):
            self.verify()
            if target in self.sources:
                backup = target.with_name(f"{target.name}.before-pa-layout-{uuid4().hex}.bak")
                self.runtime._write_text(backup, self.sources[target].decode(), create_only=True)
            self.runtime._write_text(target, self.plans[target].decode(), create_only=target not in self.sources)
            if self.files.checked(target).read_bytes() != self.plans[target]:
                raise FileExistsError(f"PA destination changed during migration: {target}")
            self.sources[target] = self.plans[target]
        return len(self.plans)


def migrate(
    workspace: Path,
    *,
    metadata_dir: str = "reme_metadata",
    daily_dir: str = "daily",
    check: bool = False,
) -> int:
    """Plan all old layouts before writing; originals remain available for recovery."""
    context = ApplicationContext(workspace_dir=str(workspace.resolve()), metadata_dir=metadata_dir, daily_dir=daily_dir)
    runtime = PaRuntime(app_context=context)
    migration = _Migration(runtime)
    files = runtime.tasks
    legacy_config = files.checked(runtime.cron_path.with_name("tasks.json"))
    config_source = runtime.cron_path if runtime.cron_path.exists() else legacy_config
    files.parse(migration.read(config_source).decode())
    migration.plan(config_source, runtime.cron_path)
    old_inbox = runtime._metadata_file("proactive_inbox.txt")
    if old_inbox.exists() and not runtime.proactive_inbox_path.exists():
        migration.plan(old_inbox, runtime.proactive_inbox_path)
    for old, new in migration.renamed.items():
        source = runtime._metadata_file(old)
        if source.exists():
            migration.plan(source, new)
    legacy_root = files.checked(runtime.workspace_path / "reme_cron")
    if legacy_root.exists():
        for source in sorted(legacy_root.rglob("*")):
            files.checked(source)
            if source.is_file():
                relative = source.relative_to(runtime.workspace_path).as_posix()
                migration.plan(source, runtime.workspace_path / migration.relocate(relative))
    daily_root = files.checked(runtime.workspace_path / context.app_config.daily_dir)
    for directory in sorted(daily_root.glob("????-??-??")):
        try:
            date.fromisoformat(directory.name)
        except ValueError:
            continue
        for source in sorted(files.checked(directory / "cron").glob("*/*.md")):
            if not re.fullmatch(r"(?:[01][0-9]|2[0-3])[0-5][0-9][0-5][0-9]", source.stem):
                continue
            post = frontmatter.loads(migration.read(source).decode())
            if post.get("task_id") != source.parent.name or not post.content.strip():
                raise ValueError(f"Invalid legacy cron result: {source}")
            target = runtime.workspace_path / migration.relocate(source.relative_to(runtime.workspace_path).as_posix())
            migration.plan(source, target)
    for source in (runtime.temp_tasks.path, runtime.proactive_context_path):
        if source.exists():
            migration.plan(source, source)
    # Publication of the canonical journal follows every copied result and rewritten reference.
    if not runtime.chat_inbox_path.exists() and runtime.chat_inbox_path not in migration.plans:
        migration.plans[runtime.chat_inbox_path] = b""
    return migration.publish(check)


def main() -> None:
    """Run against a stopped service; --check validates without changing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="ReMe workspace; stop the owning service first")
    parser.add_argument("--metadata-dir", default="reme_metadata")
    parser.add_argument("--daily-dir", default="daily")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    count = migrate(args.workspace, metadata_dir=args.metadata_dir, daily_dir=args.daily_dir, check=args.check)
    print(f"{'Planned' if args.check else 'Migrated'} {count} PA files; originals and in-place backups are retained")


if __name__ == "__main__":
    main()
