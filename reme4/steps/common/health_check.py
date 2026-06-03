"""Concise health snapshot of ReMe runtime components."""

import sys
from collections.abc import Mapping

import numpy as np

from ..base_step import BaseStep
from ... import __version__
from ...components import R
from ...enumeration import ComponentEnum


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


def _deep_size(obj, _seen: set | None = None) -> int:
    """Recursive sizeof. Uses ndarray.nbytes; walks Mappings, sequences, __dict__."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) + sys.getsizeof(obj)

    size = sys.getsizeof(obj)
    if isinstance(obj, (str, bytes, bytearray, int, float, bool, type(None))):
        extra = 0
    elif isinstance(obj, Mapping):
        extra = sum(_deep_size(k, _seen) + _deep_size(v, _seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        extra = sum(_deep_size(item, _seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        extra = _deep_size(vars(obj), _seen)
    elif hasattr(obj, "__slots__"):
        extra = sum(_deep_size(getattr(obj, s), _seen) for s in obj.__slots__ if hasattr(obj, s))
    else:
        extra = 0
    return size + extra


def _mb_str(*objs) -> str:
    """Sum deep size of objs and format as 'X.XX MB'."""
    seen: set = set()
    total = sum(_deep_size(o, seen) for o in objs)
    return f"{total / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Per-component status collectors
# ---------------------------------------------------------------------------


def _embedding_status(comp) -> dict:
    cache = getattr(comp, "_embedding_cache", {}) or {}
    try:
        dims = comp.dimensions
    except Exception:
        dims = None
    return {
        "is_started": comp.is_started,
        "is_healthy": getattr(comp, "is_healthy", None),
        "model_name": getattr(comp, "model_name", None),
        "dimensions": dims,
        "cache_size": len(cache),
        "memory": _mb_str(cache),
    }


def _file_graph_nx_status(comp, graph) -> dict:
    """Networkx backend: virtuals are nodes without a 'node' payload."""
    n_real = sum(1 for _, d in graph.nodes(data=True) if "node" in d)
    return {
        "is_started": comp.is_started,
        "n_nodes": n_real,
        "n_edges": graph.number_of_edges(),
        "n_virtual": graph.number_of_nodes() - n_real,
        "memory": _mb_str(graph),
    }


def _file_graph_local_status(comp) -> dict:
    """Local backend: nodes / inverse edges / pending edges held as separate dicts."""
    nodes = getattr(comp, "_nodes", {}) or {}
    inverse = getattr(comp, "_inverse", {}) or {}
    pending = getattr(comp, "_pending", {}) or {}
    return {
        "is_started": comp.is_started,
        "n_nodes": len(nodes),
        "n_edges": sum(len(s) for s in inverse.values()),
        "n_pending": sum(len(s) for s in pending.values()),
        "memory": _mb_str(nodes, inverse, pending),
    }


def _file_graph_status(comp) -> dict:
    graph = getattr(comp, "_graph", None)
    return _file_graph_nx_status(comp, graph) if graph is not None else _file_graph_local_status(comp)


def _file_store_status(comp) -> dict:
    chunks = getattr(comp, "file_chunks", {}) or {}
    return {
        "is_started": comp.is_started,
        "n_chunks": len(chunks),
        "n_chunks_with_embedding": sum(1 for c in chunks.values() if getattr(c, "embedding", None) is not None),
        "memory": _mb_str(chunks),
    }


def _keyword_index_status(comp) -> dict:
    vocab = getattr(comp, "vocab", {}) or {}
    return {
        "is_started": comp.is_started,
        "n_docs": getattr(comp, "n_docs", None),
        "vocab_size": len(vocab),
        "memory": _mb_str(
            vocab,
            getattr(comp, "inverted_index", {}) or {},
            getattr(comp, "doc_meta", {}) or {},
            getattr(comp, "_idf_cache", {}) or {},
        ),
    }


_HANDLERS = {
    ComponentEnum.EMBEDDING_STORE: _embedding_status,
    ComponentEnum.FILE_GRAPH: _file_graph_status,
    ComponentEnum.FILE_STORE: _file_store_status,
    ComponentEnum.KEYWORD_INDEX: _keyword_index_status,
}


# ---------------------------------------------------------------------------
# Health rules and step entry point
# ---------------------------------------------------------------------------


def _is_healthy(ctype: ComponentEnum, status: dict) -> bool:
    """Unstarted = unhealthy; embedding model also requires is_healthy != False."""
    if not status.get("is_started"):
        return False
    if ctype is ComponentEnum.EMBEDDING_STORE and status.get("is_healthy") is False:
        return False
    return True


def _collect_components(app_context) -> tuple[dict, bool]:
    """Walk every registered component type and produce {type: {name: status}}, plus overall flag."""
    components: dict = {}
    healthy = True
    for ctype, handler in _HANDLERS.items():
        bucket = {}
        for name, comp in app_context.components.get(ctype, {}).items():
            status = handler(comp)
            bucket[name] = status
            if not _is_healthy(ctype, status):
                healthy = False
        components[ctype.value] = bucket
    return components, healthy


@R.register("health_check_step")
class HealthCheckStep(BaseStep):
    """Collect a concise health snapshot of the relevant components."""

    async def execute(self):
        assert self.context is not None

        if self.app_context is not None:
            components, healthy = _collect_components(self.app_context)
        else:
            components, healthy = {}, True

        health = {"version": __version__, "healthy": healthy, "components": components}
        self.logger.info(f"[{self.name}] health collected: {health}")

        emoji = "✅" if healthy else "❌"
        label = "healthy" if healthy else "unhealthy"
        self.context.response.answer = f"{emoji} ReMe v{__version__} - {label}"
        self.context.response.metadata["health"] = health
        return self.context.response
