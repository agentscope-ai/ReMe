"""Report ReMe runtime memory usage."""

import sys
from collections.abc import Mapping
from types import ModuleType

import numpy as np
import psutil

from ..base_step import BaseStep
from ...components import BaseComponent, R
from ...enumeration import ComponentEnum

_SKIPPED_COMPONENT_ATTRIBUTES = {"app_context", "logger"}
_TRACKED_COMPONENT_TYPES = {
    ComponentEnum.EMBEDDING_STORE,
    ComponentEnum.FILE_GRAPH,
    ComponentEnum.FILE_STORE,
    ComponentEnum.KEYWORD_INDEX,
}


def _format_bytes(size: int) -> str:
    """Format a byte count using binary units."""
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units[:-1]:
        if abs(value) < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} {units[-1]}"


def _component_size(obj: object) -> int:
    """Estimate the memory owned by one component's Python object graph.

    References to the application context, logger, and other components are
    excluded so shared application state is not charged repeatedly.
    """
    seen: set[int] = set()
    root_id = id(obj)

    def walk(value: object) -> int:  # pylint: disable=too-many-return-statements
        value_id = id(value)
        if value_id in seen:
            return 0
        seen.add(value_id)

        if isinstance(value, BaseComponent) and value_id != root_id:
            return 0
        if isinstance(value, (type, ModuleType)):
            return 0
        if isinstance(value, np.ndarray):
            # ndarray.__sizeof__ normally includes its owned data buffer.  A
            # view may be smaller, so retain at least the visible buffer size.
            return max(sys.getsizeof(value), int(value.nbytes))

        size = sys.getsizeof(value)
        if isinstance(
            value,
            (str, bytes, bytearray, memoryview, int, float, bool, complex, type(None)),
        ):
            return size
        if isinstance(value, Mapping):
            return size + sum(walk(key) + walk(item) for key, item in value.items())
        if isinstance(value, (list, tuple, set, frozenset)):
            return size + sum(walk(item) for item in value)
        if hasattr(value, "__dict__"):
            attributes = vars(value)
            if id(attributes) in seen:
                return size
            seen.add(id(attributes))
            return (
                size
                + sys.getsizeof(attributes)
                + sum(
                    walk(key) + walk(item)
                    for key, item in attributes.items()
                    if key not in _SKIPPED_COMPONENT_ATTRIBUTES
                )
            )
        slots = getattr(value, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        return size + sum(walk(getattr(value, slot)) for slot in slots if hasattr(value, slot))

    return walk(obj)


def _collect_memory(app_context) -> dict:
    """Collect per-component estimates, their sum, and process RSS."""
    components: dict[str, dict[str, dict[str, int | str]]] = {}
    total = 0
    if app_context is not None:
        for component_type in sorted(_TRACKED_COMPONENT_TYPES, key=lambda item: item.value):
            group = {}
            for name, component in sorted(app_context.components.get(component_type, {}).items()):
                size = _component_size(component)
                group[name] = {"bytes": size, "human": _format_bytes(size)}
                total += size
            if group:
                components[component_type.value] = group

    rss = psutil.Process().memory_info().rss
    return {
        "components": components,
        "components_total_bytes": total,
        "components_total": _format_bytes(total),
        "process_rss_bytes": rss,
        "process_rss": _format_bytes(rss),
    }


def _format_status(memory: dict) -> str:
    """Build the human-readable CLI response."""
    lines = ["Memory (estimated component object size)"]
    for component_type, group in memory["components"].items():
        for name, usage in group.items():
            lines.append(f"  {component_type}:{name}  {usage['human']}")
    lines.extend(
        [
            f"  Components total  {memory['components_total']}",
            f"  Process RSS       {memory['process_rss']}",
        ],
    )
    return "\n".join(lines)


@R.register("status_step")
class StatusStep(BaseStep):
    """Report per-component memory estimates and process RSS."""

    async def execute(self):
        assert self.context is not None

        memory = _collect_memory(self.app_context)
        self.context.response.answer = _format_status(memory)
        self.context.response.metadata["status"] = {"memory": memory}
        return self.context.response
