"""Browse Proactive's input files without invoking any consuming agent tools."""

# pylint: disable=protected-access

import asyncio
from hashlib import sha256
import json
import re

from .task_results import TaskResultsStep
from ...file_io._path import _check_path_permission
from ....components import R
from ....schema.pa_inbox import PaInboxDetail, PaInboxItem, PaInboxPage


def _timestamp(value):
    return value.isoformat() if hasattr(value, "isoformat") else str(value) if value is not None else None


def _detail(*, content, received_at=None, **values):
    item = PaInboxItem(preview=content[:240], **values)
    return PaInboxDetail(
        item=item,
        content=content,
        received_at=received_at,
    )


@R.register("proactive_inbox_view_step")
class ProactiveInboxViewStep(TaskResultsStep):
    """A derived view only: never writes files, receipts or consumption cursors."""

    def _tasks(self, warnings):
        try:
            return {task.id: task for task in self.pa.tasks.load()}
        except (OSError, ValueError) as exc:
            warnings.append(f"Task definitions: {exc}")
            return None

    def _task_detail(self, path, tasks):
        post = self._post(path)
        task = (tasks or {}).get(post["task_id"])
        raw_status = post.get("status")
        status = (
            {"unread": "pending", "read": "received", "superseded": "superseded"}.get(raw_status, "unknown")
            if isinstance(raw_status, str)
            else "unknown"
        )
        if status == "pending" and task is None:
            status = "unknown" if tasks is None else "removed"
        relative = self.to_workspace_relative(path)
        return _detail(
            id=f"task:{relative}",
            kind="task_result",
            title=task.query[:160] if task else post["task_id"],
            content=post.content,
            status=status,
            task_id=post["task_id"],
            path=relative,
            created_at=_timestamp(post.get("created_at")),
            user_read_at=_timestamp(post.get("user_read_at")),
        )

    def _task_documents(self, tasks, warnings):
        files = self.pa.tasks
        items, task_ids = [], []
        if not files.root_path.exists():
            return items, task_ids
        for root in sorted(files.root_path.iterdir()):
            if not root.is_dir():
                continue
            try:
                paths = files.paths(root.name)
            except (OSError, ValueError) as exc:
                warnings.append(f"Task {root.name}: {exc}")
                continue
            if paths:
                task_ids.append(root.name)
            for path in paths:
                try:
                    item = self._task_detail(path, tasks).item
                    items.append(item)
                    if item.status == "unknown":
                        warnings.append(f"Unknown task consumption state: {item.path}")
                except (OSError, ValueError) as exc:
                    warnings.append(f"Task result {self.to_workspace_relative(path)}: {exc}")
        return items, task_ids

    @staticmethod
    def _body(section, record):
        body = section.partition("\n")[2].strip()
        header = f"{record['created_at']} · Main Agent"
        if body.startswith(header + "\n"):
            body = body[len(header) :].strip()
        return body

    def _instructions(self, warnings, entry_id=None):
        inbox = self.pa.inbox
        records = inbox.receipts()
        try:
            text = self.pa.tasks.read(self.pa.proactive_inbox_path)
        except (OSError, ValueError) as exc:
            warnings.append(f"Instruction source: {exc}")
            text = None
        raw = (text or "").encode("utf-8")
        consumed = 0
        try:
            if inbox.cursor_path.exists():
                cursor = json.loads(self.pa.tasks.read(inbox.cursor_path))
                if not isinstance(cursor, dict):
                    raise ValueError("Invalid proactive inbox cursor")
                offset = cursor["bytes"]
                if type(offset) is not int or offset < 0:
                    raise ValueError("Invalid proactive inbox cursor")
                if offset <= len(raw) and sha256(raw[:offset]).hexdigest() == cursor.get("sha256"):
                    consumed = len(raw[:offset].decode("utf-8"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Instruction consumption state: {exc}")

        # Only receipt-owned headings delimit entries; ordinary Markdown headings remain content.
        headings = [
            match
            for match in re.finditer(r"^## ([A-Za-z0-9_-]{1,128})[ \t]*\r?$", text or "", re.MULTILINE)
            if match[1] in records
        ]
        sections = {}
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            section = text[heading.start() : end].strip()
            if heading[1] in sections:
                raise ValueError("Instruction source contains duplicate request headings")
            sections[heading[1]] = (section, heading.start() + len(section))
        prefix_text = (text or "")[: headings[0].start() if headings else len(text or "")]
        prefix = prefix_text.strip()
        if prefix and entry_id in {None, "instruction-file"}:
            end = len(prefix_text.rstrip())
            yield _detail(
                id="instruction-file",
                kind="instruction",
                title=prefix.splitlines()[0].lstrip("# ")[:160],
                content=prefix,
                status="received" if end <= consumed else "pending",
                source_modified=True,
            )
        for identifier, record in records.items():
            if entry_id is not None and entry_id != f"instruction:{identifier}":
                continue
            section, end = sections.get(identifier, (None, 0))
            original = inbox._entry(record).strip()
            modified = section is not None and section != original
            receipt_status = record["status"]
            status = {
                "staging": "pending",
                "queued": "pending",
                "consumed": "received",
            }.get(receipt_status, "unknown")
            if status == "unknown":
                warnings.append(f"Unknown instruction receipt state: {identifier}")
            if text is None and status in {"pending", "received"}:
                status = "unknown"
            elif section is None and status in {"pending", "received"} and receipt_status != "staging":
                status = "removed"
            elif section is not None:
                status = "received" if end <= consumed else "pending"
            body = self._body(section or original, record)
            yield _detail(
                id=f"instruction:{identifier}",
                kind="instruction",
                title=body.splitlines()[0][:160] if body else "Main Agent",
                content=body,
                status=status,
                created_at=_timestamp(record.get("created_at")),
                source_present=section is not None if text is not None else None,
                source_modified=modified,
                receipt_status=receipt_status,
                received_at=_timestamp(record.get("consumed_at")),
            )

    def _page(self, kind, state_filter, task_id, offset, limit):
        warnings = []
        tasks = self._tasks(warnings)
        try:
            items, task_ids = self._task_documents(tasks, warnings)
        except (OSError, ValueError) as exc:
            warnings.append(f"Task results: {exc}")
            items, task_ids = [], []
        try:
            items.extend(document.item for document in self._instructions(warnings))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Instruction receipts: {exc}")
        pending = sum(item.status == "pending" for item in items)
        selected = [
            item
            for item in items
            if (
                kind in {"all", item.kind}
                and (not task_id or item.task_id == task_id)
                and (
                    state_filter == "all"
                    or state_filter == "pending"
                    and item.status == "pending"
                    or state_filter == "unread"
                    and item.kind == "task_result"
                    and item.user_read_at is None
                )
            )
        ]
        selected.sort(key=lambda item: (item.status == "pending", item.created_at or "", item.id), reverse=True)
        return PaInboxPage(
            items=selected[offset : offset + limit],
            total=len(selected),
            offset=offset,
            limit=limit,
            task_ids=task_ids,
            pending=pending,
            complete=not warnings,
            warnings=warnings[:20],
        )

    async def execute(self):
        files, inbox = self.pa.tasks, self.pa.inbox
        for path in (
            files.root_path,
            self.pa.cron_path,
            self.pa.proactive_inbox_path,
            inbox.cursor_path,
            inbox.receipts_path,
        ):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to browse proactive inbox")
        operation = self.context.get("action", "list")
        async with files.lock, inbox.lock:
            if operation == "list":
                kind, state_filter = self.context.get("kind", "all"), self.context.get("filter", "all")
                task_id = self.context.get("task_id")
                offset, limit = self.context.get("offset", 0), self.context.get("limit", 20)
                if kind not in {"all", "task_result", "instruction"} or state_filter not in {
                    "all",
                    "pending",
                    "unread",
                }:
                    raise ValueError("Invalid inbox kind or filter")
                if state_filter == "unread" and kind != "task_result" or task_id and kind != "task_result":
                    raise ValueError("User-unread and task filters require task_result kind")
                if task_id:
                    files.task_path(task_id)
                if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
                    raise ValueError("offset must be nonnegative and limit must be 1-100")
                result = await asyncio.to_thread(self._page, kind, state_filter, task_id, offset, limit)
            elif operation == "read":
                identifier = self.context.get("entry_id")
                if not isinstance(identifier, str) or not identifier or len(identifier) > 8192:
                    raise ValueError("An inbox entry_id is required")
                if identifier.startswith("task:"):
                    path = self._result_path(identifier[5:])
                    warnings = []
                    result = await asyncio.to_thread(lambda: self._task_detail(path, self._tasks(warnings)))
                    result.warnings = warnings
                else:
                    warnings = []
                    result = await asyncio.to_thread(lambda: next(self._instructions(warnings, identifier), None))
                    if result is None:
                        raise ValueError("Inbox entry no longer exists")
                    result.warnings = warnings
            else:
                raise ValueError("action must be list or read")
        self.context.response.answer = result.model_dump(mode="json")
        return self.context.response
