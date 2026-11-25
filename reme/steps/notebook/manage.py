"""Two small jobs for conversational Todo and Idea management."""

from abc import abstractmethod
from datetime import datetime, timezone

from ..base_step import BaseStep
from ...utils.files.path_lock import get_path_lock
from ...core.registry import R
from ...schema import Idea, Todo, TodoProgress
from ...schema.notebook import IdeaRequest, TodoRequest
from ._store import RecordStore


class _RecordStep(BaseStep):
    """Keep invocation state local; the filesystem owns all durable records."""

    kind = ""
    request_model = None
    automatic_fields = frozenset()

    async def execute(self):
        arguments = {
            key: value
            for key, value in self.context.data.items()
            if key not in {"metadata", "_allowed_paths", "_agent_name", "_inbox_turn", "_automatic_notebook"}
        }
        request = self.request_model.model_validate(arguments)
        self._check_automatic_request(request)
        async with await get_path_lock(self.workspace_path.resolve() / ".notebook.lock"):
            self.context.response.answer = await self.handle_request(request)
        return self.context.response

    def _check_automatic_request(self, request):
        if not self.context.get("_automatic_notebook") or request.action in {"list", "list_projects", "get"}:
            return
        fields = request.model_fields_set - {"action", "id"}
        if request.action != "update" or not fields <= self.automatic_fields:
            raise PermissionError("Automatic notebook tools can only update supported fields")
        if "status" in fields and request.status != "in_progress":
            raise PermissionError("Automatic Todo updates can only set the status to in_progress")

    async def handle_request(self, request):
        """Run the shared record actions under the notebook lock."""
        store = self._record_store(self.kind)
        await store.load(archives=request.action != "list" or request.scope != "active")
        if request.action == "list":
            return self.list_records(store, request)
        if request.action == "get":
            return store.get(request.id).model_dump(mode="json", exclude_none=True)
        original = store.get(request.id) if request.id else None
        if self.context.get("_automatic_notebook") and original is not None and original.closed_at is not None:
            raise PermissionError("Automatic notebook updates require an active record")
        record = await self.change_record(original, request)
        if record != original:
            await store.put(record)
        return record.model_dump(mode="json", exclude_none=True)

    def _record_store(self, kind: str) -> RecordStore:
        return RecordStore(
            self.workspace_path,
            self.config_value("notebook_dir"),
            kind,
            self.context.get("_allowed_paths"),
        )

    @staticmethod
    def records_in_scope(store, scope):
        """Yield records in reverse file order for the requested scope."""
        for record in reversed(store.records.values()):
            archived = store.locations[record.id].startswith("archive/")
            if scope == "all" or archived == (scope == "archive"):
                yield record

    def list_records(self, store, request):
        """Return bounded summaries; get returns the complete progress history."""
        records = self.records_in_scope(store, request.scope)
        by_creation = request.order == "created" if isinstance(request, TodoRequest) else request.scope == "active"
        if request.scope == "archive":
            records = sorted(records, key=lambda record: record.closed_at, reverse=True)
        elif by_creation:
            records = sorted(
                records,
                key=lambda record: (record.created_at is not None, record.created_at),
                reverse=True,
            )
        elif request.scope == "active":
            records = sorted(records, key=lambda record: (record.due_date is None, record.due_date))
        items = []
        for record in records:
            if "project" in request.model_fields_set and record.project != request.project:
                continue
            if isinstance(request, TodoRequest):
                if "status" in request.model_fields_set and record.status != request.status:
                    continue
                if "source_idea" in request.model_fields_set and record.source_idea != request.source_idea:
                    continue
            text = record.project + "\n" + record.text
            if isinstance(record, Todo):
                text += "\n" + "\n".join(progress.text for progress in record.progress)
                if record.close_note:
                    text += "\n" + record.close_note
            if request.query.casefold() not in text.casefold():
                continue
            value = record.model_dump(mode="json", exclude_none=True)
            if isinstance(record, Todo):
                value["progress_count"] = len(value["progress"])
                value["progress"] = value["progress"][-1:]
            items.append(value)
        return {
            "items": items[request.offset : request.offset + request.limit],
            "total": len(items),
            "offset": request.offset,
            "limit": request.limit,
        }

    @abstractmethod
    async def change_record(self, original, request):
        """Subclasses implement their few domain-specific state transitions."""


@R.register("todo_step")
class TodoStep(_RecordStep):
    """Manage tasks and discover projects across the notebook."""

    kind = "todos"
    request_model = TodoRequest
    automatic_fields = frozenset({"progress_text", "progress_time", "progress_match", "status"})

    async def handle_request(self, request):
        """Aggregate project counts across both record collections."""
        if request.action != "list_projects":
            return await super().handle_request(request)

        scope = request.scope if "scope" in request.model_fields_set else "all"
        projects = {}
        for kind, count_field in (("todos", "todo_count"), ("ideas", "idea_count")):
            store = self._record_store(kind)
            await store.load(archives=scope != "active")
            for record in self.records_in_scope(store, scope):
                project = projects.setdefault(
                    record.project,
                    {"project": record.project, "todo_count": 0, "idea_count": 0},
                )
                project[count_field] += 1
        return {"items": [projects[name] for name in sorted(projects)], "total": len(projects)}

    async def change_record(self, original, request):
        fields = {"project", "text", "status", "due_date", "source_idea", "close_note"}
        changes = request.model_dump(include=fields, exclude_unset=True)
        if request.source_idea:
            ideas = self._record_store("ideas")
            await ideas.load()
            idea = ideas.get(request.source_idea)
            if request.action == "create":
                changes.setdefault("project", idea.project)
                changes.setdefault("text", idea.text)
        record = (
            Todo(created_at=datetime.now(timezone.utc), **changes)
            if original is None
            else Todo.model_validate({**original.model_dump(), **changes})
        )
        self._update_progress(record.progress, request)
        if record.status == "closed":
            record.closed_at = record.closed_at or datetime.now(timezone.utc)
        else:
            record.closed_at = None
            record.close_note = None
        return record

    def _update_progress(self, progress, request):
        match = request.progress_match
        index = None
        if match is not None:
            matches = [i for i, note in enumerate(progress) if note == match]
            if len(matches) != 1:
                raise ValueError("Expected exactly one matching progress entry; read the Todo before trying again")
            index = matches[0]
            if request.progress_text is None:
                del progress[index]
                return
        if request.progress_text is None or (
            self.context.get("_automatic_notebook")
            and any(note.text == request.progress_text for i, note in enumerate(progress) if i != index)
        ):
            return
        time = request.progress_time or (match.time if match else None)
        values = {"time": time} if time is not None else {}
        note = TodoProgress(text=request.progress_text, **values)
        if index is None:
            progress.append(note)
        else:
            progress[index] = note


@R.register("idea_step")
class IdeaStep(_RecordStep):
    """Archive ideas explicitly while retaining their stable source identities."""

    kind = "ideas"
    request_model = IdeaRequest

    async def change_record(self, original, request):
        changes = request.model_dump(include={"project", "text"}, exclude_unset=True)
        if original is None:
            return Idea(created_at=datetime.now(timezone.utc), **changes)
        if request.action == "update":
            return Idea.model_validate({**original.model_dump(), **changes})
        record = original.model_copy()
        if request.action == "archive":
            record.closed_at = record.closed_at or datetime.now(timezone.utc)
        else:
            record.closed_at = None
        return record
