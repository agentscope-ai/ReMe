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

    async def execute(self):
        arguments = {
            key: value
            for key, value in self.context.data.items()
            if key not in {"metadata", "_allowed_paths", "_agent_name", "_inbox_turn"}
        }
        request = self.request_model.model_validate(arguments)
        async with await get_path_lock(self.workspace_path.resolve() / ".notebook.lock"):
            self.context.response.answer = await self.handle_request(request)
        return self.context.response

    async def handle_request(self, request):
        """Run the shared record actions under the notebook lock."""
        store = self._record_store(self.kind)
        await store.load(archives=request.action != "list" or request.scope != "active")
        if request.action == "list":
            return self.list_records(store, request)
        if request.action == "get":
            return store.get(request.id).model_dump(mode="json", exclude_none=True)
        record = await self.change_record(store, request)
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
        items = []
        for record in self.records_in_scope(store, request.scope):
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
            if request.query.casefold() not in text.casefold():
                continue
            value = record.model_dump(mode="json", exclude_none=True)
            if isinstance(record, Todo):
                value["progress"] = value["progress"][-1:]
                value["progress_count"] = len(record.progress)
            items.append(value)
        return {
            "items": items[request.offset : request.offset + request.limit],
            "total": len(items),
            "offset": request.offset,
            "limit": request.limit,
        }

    @abstractmethod
    async def change_record(self, store, request):
        """Subclasses implement their few domain-specific state transitions."""


@R.register("todo_step")
class TodoStep(_RecordStep):
    """Manage tasks and discover projects across the notebook."""

    kind = "todos"
    request_model = TodoRequest

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

    async def change_record(self, store, request):
        fields = {"project", "text", "status", "due_date", "source_idea"}
        changes = request.model_dump(include=fields, exclude_unset=True)
        if request.source_idea:
            ideas = self._record_store("ideas")
            await ideas.load()
            idea = ideas.get(request.source_idea)
            if request.action == "create":
                changes.setdefault("project", idea.project)
                changes.setdefault("text", idea.text)
        record = Todo(**changes) if request.action == "create" else store.get(request.id).model_copy(deep=True)
        if request.action == "update":
            record = Todo.model_validate({**record.model_dump(), **changes})
        if request.progress_text:
            note = {"text": request.progress_text}
            if request.progress_time is not None:
                note["time"] = request.progress_time
            record.progress.append(TodoProgress(**note))
        if record.status in {"done", "cancelled"}:
            record.closed_at = record.closed_at or datetime.now(timezone.utc)
        else:
            record.closed_at = None
        return record


@R.register("idea_step")
class IdeaStep(_RecordStep):
    """Archive ideas explicitly while retaining their stable source identities."""

    kind = "ideas"
    request_model = IdeaRequest

    async def change_record(self, store, request):
        changes = request.model_dump(include={"project", "text"}, exclude_unset=True)
        if request.action == "create":
            return Idea(**changes)
        record = store.get(request.id).model_copy(deep=True)
        if request.action == "update":
            record = Idea.model_validate({**record.model_dump(), **changes})
        elif request.action == "archive":
            record.closed_at = record.closed_at or datetime.now(timezone.utc)
        elif request.action == "restore":
            record.closed_at = None
        return record
