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
            store = self._record_store(self.kind)
            await store.load(archives=request.action != "list" or request.scope != "active")
            if request.action == "list":
                answer = self.list_records(store, request)
            elif request.action == "get":
                answer = store.get(request.id).model_dump(mode="json", exclude_none=True)
            else:
                record = await self.change_record(store, request)
                await store.put(record)
                answer = record.model_dump(mode="json", exclude_none=True)
        self.context.response.answer = answer
        return self.context.response

    def _record_store(self, kind: str) -> RecordStore:
        return RecordStore(
            self.workspace_path,
            self.config_value("notebook_dir"),
            kind,
            self.context.get("_allowed_paths"),
        )

    @staticmethod
    def list_records(store, request):
        """Return bounded summaries; get returns the complete progress history."""
        items = []
        for record in reversed(list(store.records.values())):
            archived = store.locations[record.id].startswith("archive/")
            if request.scope != "all" and archived != (request.scope == "archive"):
                continue
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
    """Patch task fields and append progress without replacing previous notes."""

    kind = "todos"
    request_model = TodoRequest

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
