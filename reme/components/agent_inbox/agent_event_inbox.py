"""Per-Agent inbox directories backed by the local unread/read logs."""

from datetime import datetime
import json

from pydantic import BaseModel

from .local_agent_inbox import LocalAgentInbox
from ..component_registry import R
from ...schema.agent_runtime import validate_agent_name
from ...schema.application_config import ApplicationConfig
from ...schema.event import Event


@R.register("event")
class AgentEventInbox(LocalAgentInbox):
    """Adapt storage locations without owning Agent definitions or notification policy."""

    def __init__(self, **kwargs):
        context = kwargs.get("app_context")
        directory = context.app_config.reme_agent_dir if context else ApplicationConfig().reme_agent_dir
        super().__init__(directory=directory, **kwargs)

    @staticmethod
    def _normalize_name(value):
        return validate_agent_name(value)

    def _names(self):
        return [
            self._normalize_name(path.parent.name)
            for path in sorted(self._root.glob("*/inbox"))
            if path.parent.name != "Remy"
        ]

    def _inbox_path(self, name):
        return self._root / name / "inbox"

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
