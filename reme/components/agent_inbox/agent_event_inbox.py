"""Per-Agent inbox directories backed by the local unread/read logs."""

from datetime import datetime
import json

from pydantic import BaseModel

from .local_agent_inbox import LocalAgentInbox
from ..component_registry import R
from ...schema.agent_runtime import validate_agent_name
from ...schema.application_config import ApplicationConfig
from ...schema.event import Event
from ...utils.async_utils import finish_cleanup


@R.register("event")
class AgentEventInbox(LocalAgentInbox):
    """Adapt storage locations without owning Agent definitions or notification policy."""

    def __init__(self, **kwargs):
        context = kwargs.get("app_context")
        directory = context.app_config.reme_agent_dir if context else ApplicationConfig().reme_agent_dir
        super().__init__(directory=directory, rebuildable=["Remy"], **kwargs)
        self.paths = {}

    @staticmethod
    def _normalize_name(value):
        return validate_agent_name(value)

    def _names(self):
        self.paths.clear()
        for pattern in ("*/inbox", "tmp/*/inbox"):
            for path in sorted(self._root.glob(pattern)):
                name = self._normalize_name(path.parent.name)
                if name in self.paths:
                    raise ValueError(f"duplicate Agent inbox directories: {name}")
                self.paths[name] = path
        return list(self.paths)

    def _inbox_path(self, name):
        return self.paths.setdefault(name, self._root / name / "inbox")

    def _fingerprint(self, item):
        value = {**item, "metadata": dict(item["metadata"])} if "metadata" in item else item
        if "metadata" in value:
            value["metadata"].pop("_received_at", None)
        return super()._fingerprint(value)

    async def mount(self, name, directory):
        """Mount a Runtime-selected directory under the permanent or temporary root."""
        self._normalize_name(name)
        Event.validate_paths([directory])
        path = self._workspace / directory
        if path not in {self._root / name / "inbox", self._root / "tmp" / name / "inbox"}:
            raise ValueError("directory does not match the Agent inbox storage layout")
        async with self._changed:
            if name in self.paths and self.paths[name] != path:
                raise ValueError("cannot change a mounted inbox's storage directory")
            self.paths[name] = path
            await finish_cleanup(self._mount(name))

    async def forget(self, name):
        await super().forget(name)
        self.paths.pop(name, None)

    @staticmethod
    def _validate(value):
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return Event.model_validate(value).model_dump(mode="json")

    def _item(self, raw):
        value = json.loads(raw)
        result = self._validate(value)
        if result.keys() != value.keys():
            raise ValueError("persisted inputs must contain every item field")
        return result

    @staticmethod
    def _sort_key(item):
        return datetime.fromisoformat(item["event_ts"]), item["id"]
