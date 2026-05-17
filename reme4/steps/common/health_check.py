"""Return a concise health check snapshot of ReMe runtime components."""

import sys
from collections.abc import Mapping

import numpy as np

from ..base_step import BaseStep
from ... import __version__
from ...components import R
from ...enumeration import ComponentEnum


def _deep_size(obj, _seen: set | None = None) -> int:
    """Recursive sizeof; uses ndarray.nbytes for numpy and walks containers / __dict__."""
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) + sys.getsizeof(obj)

    size = sys.getsizeof(obj)
    if isinstance(obj, (str, bytes, bytearray, int, float, bool, type(None))):
        return size
    if isinstance(obj, Mapping):
        size += sum(_deep_size(k, _seen) + _deep_size(v, _seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, _seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += _deep_size(vars(obj), _seen)
    elif hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                size += _deep_size(getattr(obj, slot), _seen)
    return size


def _mb_str(*objs) -> str:
    """Return summed deep size of objs formatted as 'X.XX MB'."""
    seen: set = set()
    total = sum(_deep_size(o, seen) for o in objs)
    return f"{total / (1024 * 1024):.2f} MB"


def _embedding_status(comp) -> dict:
    return {
        "is_started": comp.is_started,
        "is_healthy": getattr(comp, "is_healthy", None),
        "model_name": getattr(comp, "model_name", None),
        "dimensions": getattr(comp, "dimensions", None),
        "cache_size": len(getattr(comp, "_embedding_cache", {}) or {}),
        "memory": _mb_str(getattr(comp, "_embedding_cache", {}) or {}),
    }


def _file_graph_status(comp) -> dict:
    nodes = getattr(comp, "_nodes", {}) or {}
    inverse = getattr(comp, "_inverse", {}) or {}
    pending = getattr(comp, "_pending", {}) or {}
    return {
        "is_started": comp.is_started,
        "n_nodes": len(nodes),
        "n_edges": sum(len(s) for s in inverse.values()),
        "memory": _mb_str(nodes, inverse, pending),
    }


def _file_store_status(comp) -> dict:
    chunks = getattr(comp, "file_chunks", {}) or {}
    return {
        "is_started": comp.is_started,
        "n_chunks": len(chunks),
        "n_chunks_with_embedding": sum(1 for c in chunks.values() if getattr(c, "embedding", None) is not None),
        "memory": _mb_str(chunks),
    }


def _file_watcher_status(comp) -> dict:
    bg = getattr(comp, "_background_task", None)
    return {
        "is_started": comp.is_started,
        "background_running": bool(bg and not bg.done()),
        "watch_paths": [str(p) for p in (getattr(comp, "watch_paths", []) or [])],
    }


def _keyword_index_status(comp) -> dict:
    return {
        "is_started": comp.is_started,
        "n_docs": getattr(comp, "n_docs", None),
        "vocab_size": len(getattr(comp, "vocab", {}) or {}),
        "memory": _mb_str(
            getattr(comp, "vocab", {}) or {},
            getattr(comp, "inverted_index", {}) or {},
            getattr(comp, "doc_meta", {}) or {},
            getattr(comp, "_idf_cache", {}) or {},
        ),
    }


_HANDLERS = {
    ComponentEnum.EMBEDDING_MODEL: _embedding_status,
    ComponentEnum.FILE_GRAPH: _file_graph_status,
    ComponentEnum.FILE_STORE: _file_store_status,
    ComponentEnum.FILE_WATCHER: _file_watcher_status,
    ComponentEnum.KEYWORD_INDEX: _keyword_index_status,
}


def _is_status_healthy(ctype: ComponentEnum, status: dict) -> bool:
    """Per-component health rule. Unstarted = unhealthy; type-specific extras checked."""
    if not status.get("is_started"):
        return False
    if ctype is ComponentEnum.EMBEDDING_MODEL and status.get("is_healthy") is False:
        return False
    if ctype is ComponentEnum.FILE_WATCHER and not status.get("background_running"):
        return False
    return True


@R.register("health_check_step")
class HealthCheckStep(BaseStep):
    """Collect a concise health check snapshot of the relevant components."""

    async def execute(self):
        assert self.context is not None

        components: dict = {}
        healthy = True
        if self.app_context is not None:
            for ctype, handler in _HANDLERS.items():
                comp_map = self.app_context.components.get(ctype, {})
                bucket = {}
                for name, comp in comp_map.items():
                    s = handler(comp)
                    bucket[name] = s
                    if not _is_status_healthy(ctype, s):
                        healthy = False
                components[ctype.value] = bucket

        health = {"version": __version__, "healthy": healthy, "components": components}
        self.logger.info(f"[{self.name}] health collected: {health}")

        status_emoji = "✅" if healthy else "❌"
        self.context.response.answer = f"{status_emoji} ReMe v{__version__} - {'healthy' if healthy else 'unhealthy'}"
        self.context.response.metadata["health"] = health
        return self.context.response
