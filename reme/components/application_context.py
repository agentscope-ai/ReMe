"""Application context: shared state container for components, jobs, and service."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..enumeration import ComponentEnum
from ..schema import ApplicationConfig

if TYPE_CHECKING:
    from .base_component import BaseComponent
    from .job import BaseJob
    from .service import BaseService


class ApplicationContext:
    """Passive state container holding parsed config and wired components.

    The Application class performs the actual wiring (registry lookups and
    component instantiation); this class only stores the results so that
    components, jobs, and the service can find each other at runtime.
    """

    def __init__(self, **kwargs):
        # Parse raw kwargs into a typed, validated config object.
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)

        # Populated by Application during initialization.
        self.service: "BaseService | None" = None
        self.components: dict[ComponentEnum, dict[str, "BaseComponent"]] = {}
        self.jobs: dict[str, "BaseJob"] = {}
        self.thread_pool: ThreadPoolExecutor | None = None

        # Application-lifetime shared state. Values remain available across Job and Step
        # invocations while this Application is running, so Jobs may keep cross-call state here.
        # This is in-memory state, not durable storage; use workspace files or a store when state
        # must survive an Application restart.
        self.metadata: dict[str, Any] = {}

        # Per-key monotonic counters organized as a nested tree. Each node holds a
        # ``value`` counter and a ``children`` dict keyed by the elements of the ``key``
        # list passed to :meth:`global_counter_next`. Sibling keys have independent
        # counters; the root node acts as the global counter for an empty key. A single
        # lock serializes all tree access and increments.
        self._counter_tree: dict[str, Any] = {"value": 0, "children": {}}
        self._counter_tree_lock = threading.Lock()

    def global_counter_next(self, key: list[str]) -> int:
        """Return the next monotonic value for ``key``, starting at 1.

        Walks the counter tree along ``key``, creating missing nodes on the way, then
        increments and returns the target node's counter. An empty ``key`` increments
        the root node, which serves as a process-wide thread-safe global counter.
        """
        with self._counter_tree_lock:
            node: dict[str, Any] = self._counter_tree
            for part in key:
                assert isinstance(part, str)
                tmp = node["children"].get(part, None)
                if tmp is None:
                    tmp = {"value": 0, "children": {}}
                    node["children"][part] = tmp
                node = tmp
            res = node["value"] + 1
            node["value"] = res
        return res
