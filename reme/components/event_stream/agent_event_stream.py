"""Per-Agent event directories backed by the local unread/read logs."""

from .local_event_stream import LocalEventStream
from ..component_registry import R
from ...schema.agent_runtime import validate_agent_id
from ...schema.application_config import ApplicationConfig
from ...schema.event import Event
from ...utils.async_utils import finish_cleanup


@R.register("agent")
class AgentEventStream(LocalEventStream):
    """Adapt storage locations without owning Agent definitions or notification policy."""

    def __init__(self, **kwargs):
        context = kwargs.get("app_context")
        directory = context.app_config.reme_agent_dir if context else ApplicationConfig().reme_agent_dir
        super().__init__(directory=directory, **kwargs)
        self.paths = {}

    @staticmethod
    def _normalize_name(value):
        return validate_agent_id(value)

    def _names(self):
        self.paths.clear()
        for pattern in ("*/events", "tmp/*/events"):
            for path in sorted(self._root.glob(pattern)):
                name = self._normalize_name(path.parent.name)
                if name in self.paths:
                    raise ValueError(f"duplicate Agent event directories: {name}")
                self.paths[name] = path
        return list(self.paths)

    def _stream_path(self, name):
        return self.paths.setdefault(name, self._root / name / "events")

    def _fingerprint(self, event):
        value = {**event, "metadata": dict(event["metadata"])}
        value["metadata"].pop("_received_at", None)
        return super()._fingerprint(value)

    async def mount(self, name, directory):
        """Mount a Runtime-selected directory under the permanent or temporary root."""
        self._normalize_name(name)
        Event.validate_paths([directory])
        path = self._workspace / directory
        if path not in {self._root / name / "events", self._root / "tmp" / name / "events"}:
            raise ValueError("directory does not match the Agent event-stream storage layout")
        async with self._changed:
            if name in self.paths and self.paths[name] != path:
                raise ValueError("cannot change a mounted stream's storage directory")
            self.paths[name] = path
            await finish_cleanup(self._mount(name))

    async def forget(self, name):
        await super().forget(name)
        self.paths.pop(name, None)
