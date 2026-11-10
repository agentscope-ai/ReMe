"""Read-only projections and explicit user-read receipts for PA files."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import frontmatter

from ...schema.pa_inbox import PaInboxDetail, PaInboxItem, PaInboxPage
from ...utils.async_utils import complete_in_thread
from ...utils.file_utils import get_path_lock

if TYPE_CHECKING:
    from .runtime import PaRuntime


def _timestamp(value):
    return value.isoformat() if hasattr(value, "isoformat") else str(value) if value is not None else None


def _detail(*, content, received_at=None, **values):
    item = PaInboxItem(preview=content[:240], **values)
    return PaInboxDetail(
        item=item,
        content=content,
        received_at=received_at,
    )


class PaViews:
    """Share document parsing between result browsing and the proactive inbox."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime

    def _result_path(self, raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError("A result path is required")
        files = self.runtime.tasks
        path = self.runtime.files.checked(self.runtime.workspace_path / raw)
        parts = path.relative_to(files.root_path).parts
        if len(parts) != 3 or path not in files.paths(parts[0]):
            raise ValueError("path must identify a published task result")
        return path

    def _list(self, task_id, unread, offset, limit):
        files = self.runtime.tasks
        if task_id:
            files.task_path(task_id)
        results, warnings, task_ids = [], [], set()
        for path, _, post in files.documents(warnings=warnings):
            task_ids.add(post["task_id"])
            if task_id and post["task_id"] != task_id or unread and post.get("user_read_at") is not None:
                continue
            results.append(
                {
                    "path": self.runtime.to_workspace_relative(path),
                    "task_id": post["task_id"],
                    "created_at": post.get("created_at"),
                    "user_read_at": post.get("user_read_at"),
                    "status": post.get("status"),
                    "preview": post.content[:240],
                },
            )
        results.sort(key=lambda item: (str(item["created_at"] or ""), item["path"]), reverse=True)
        return {
            "items": results[offset : offset + limit],
            "total": len(results),
            "offset": offset,
            "limit": limit,
            "task_ids": sorted(task_ids),
            "complete": not warnings,
            "warnings": warnings[:20],
        }

    def _tasks(self, warnings):
        try:
            return {task.id: task for task in self.runtime.tasks.load()}
        except (OSError, ValueError) as exc:
            warnings.append(f"Task definitions: {exc}")
            return None

    def _task_detail(self, path, tasks, post):
        task = (tasks or {}).get(post["task_id"])
        raw_status = post.get("status")
        status = (
            {"unread": "pending", "read": "received", "superseded": "superseded"}.get(raw_status, "unknown")
            if isinstance(raw_status, str)
            else "unknown"
        )
        if status == "pending" and task is None:
            status = "unknown" if tasks is None else "removed"
        relative = self.runtime.to_workspace_relative(path)
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
        items, task_ids = [], set()
        for path, _, post in self.runtime.tasks.documents(warnings=warnings):
            task_ids.add(post["task_id"])
            try:
                item = self._task_detail(path, tasks, post).item
                items.append(item)
                if item.status == "unknown":
                    warnings.append(f"Unknown task consumption state: {item.path}")
            except (OSError, ValueError) as exc:
                warnings.append(f"Task result {self.runtime.to_workspace_relative(path)}: {exc}")
        return items, sorted(task_ids)

    @staticmethod
    def _body(section, record):
        body = section.partition("\n")[2].strip()
        header, separator, content = body.partition("\n")
        return content.strip() if separator and header.startswith(f"{record['created_at']} · ") else body

    def _instructions(self, warnings, entry_id=None):
        inbox = self.runtime.inbox
        records = inbox.receipts()
        try:
            text = self.runtime.files.read(self.runtime.proactive_inbox_path)
        except (OSError, ValueError) as exc:
            warnings.append(f"Instruction source: {exc}")
            text = None
        raw = (text or "").encode("utf-8")
        consumed = 0
        try:
            consumed = len(raw[: inbox.consumed_bytes(raw)].decode("utf-8"))
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
            original = inbox.entry(record).strip()
            body = self._body(section or original, record)
            modified = section is not None and body != self._body(original, record)
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
            yield _detail(
                id=f"instruction:{identifier}",
                kind="instruction",
                title=body.splitlines()[0][:160] if body else "Chat Agent",
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

    async def task_results(self, data):
        """Browse validated documents or record explicit user reading."""
        operation = data.get("action", "list")
        files = self.runtime.tasks
        if operation == "list":
            offset, limit = data.get("offset", 0), data.get("limit", 50)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
                raise ValueError("offset must be nonnegative and limit must be 1-100")
            unread = data.get("unread", False)
            if type(unread) is not bool:
                raise ValueError("unread must be boolean")
            result = await asyncio.to_thread(
                self._list,
                data.get("task_id"),
                unread,
                offset,
                limit,
            )
        elif operation == "read":
            path = self._result_path(data.get("path"))
            mark_read = data.get("mark_read", False)
            if type(mark_read) is not bool:
                raise ValueError("mark_read must be boolean")
            async with files.lock:
                async with await get_path_lock(path):
                    post = await asyncio.to_thread(self.runtime.tasks.result, path)
                    if mark_read and post.get("user_read_at") is None:
                        post["user_read_at"] = self.runtime.now().isoformat()
                        await complete_in_thread(self.runtime.files.write_text, path, self.runtime.files.markdown(post))
                    result = {
                        "path": self.runtime.to_workspace_relative(path),
                        "content": frontmatter.dumps(post) + "\n",
                        "user_read_at": post.get("user_read_at"),
                    }
        else:
            raise ValueError("action must be list or read")
        return result

    async def inbox(self, data):
        """Read a consistent projection without advancing agent consumption."""
        files, inbox = self.runtime.tasks, self.runtime.inbox
        operation = data.get("action", "list")
        async with files.lock, inbox.lock:
            if operation == "list":
                kind, state_filter = data.get("kind", "all"), data.get("filter", "all")
                task_id = data.get("task_id")
                offset, limit = data.get("offset", 0), data.get("limit", 20)
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
                identifier = data.get("entry_id")
                if not isinstance(identifier, str) or not identifier or len(identifier) > 8192:
                    raise ValueError("An inbox entry_id is required")
                if identifier.startswith("task:"):
                    path = self._result_path(identifier[5:])
                    warnings = []
                    result = await asyncio.to_thread(
                        lambda: self._task_detail(path, self._tasks(warnings), self.runtime.tasks.result(path)),
                    )
                    result.warnings = warnings
                else:
                    warnings = []
                    result = await asyncio.to_thread(lambda: next(self._instructions(warnings, identifier), None))
                    if result is None:
                        raise ValueError("Inbox entry no longer exists")
                    result.warnings = warnings
            else:
                raise ValueError("action must be list or read")
        return result.model_dump(mode="json")
