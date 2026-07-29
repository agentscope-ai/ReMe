"""Thread-safe monotonic counter tree utility for shared application state."""

import copy
import threading
from typing import Any

COUNTER_TREE_KEY = "_counter_tree"
COUNTER_LOCK_KEY = "_counter_tree_lock"


def global_counter_add(metadata: dict[str, Any], key: list[str], val: int) -> int:
    """Fetch-and-add: return the old value for ``key``, then add ``val`` to it.

    Walks the counter tree stored in ``metadata`` along ``key``, creating
    missing nodes on the way, then returns the target node's current counter
    value and adds ``val`` to it. Counters start at 0, so the first call
    returns 0. An empty ``key`` targets the root node, which serves as a
    process-wide thread-safe global counter.

    The counter tree (``{"value": 0, "children": {}}``) and its
    :class:`threading.Lock` are expected to live in ``metadata`` under
    :data:`COUNTER_TREE_KEY` and :data:`COUNTER_LOCK_KEY` respectively.
    If they are missing they are created lazily so the function is safe to
    call with a plain ``dict``.
    """
    lock = metadata.get(COUNTER_LOCK_KEY)
    if lock is None:
        lock = threading.Lock()
        metadata[COUNTER_LOCK_KEY] = lock

    with lock:
        tree = metadata.get(COUNTER_TREE_KEY)
        if tree is None:
            tree = {"value": 0, "children": {}}
            metadata[COUNTER_TREE_KEY] = tree

        node: dict[str, Any] = tree
        for part in key:
            assert isinstance(part, str)
            tmp = node["children"].get(part, None)
            if tmp is None:
                tmp = {"value": 0, "children": {}}
                node["children"][part] = tmp
            node = tmp
        res = node["value"]
        node["value"] = res + val
    return res


def global_counter_inc(metadata: dict[str, Any], key: list[str]) -> int:
    """Fetch-and-increment: return the old value for ``key``, then add 1.

    Counters start at 0, so the first call returns 0. See
    :func:`global_counter_add` for details on the counter tree layout.
    """
    return global_counter_add(metadata, key, 1)


def global_counter_get(metadata: dict[str, Any], key: list[str]) -> int:
    """Return the current value for ``key`` without modifying the tree.

    Unlike :func:`global_counter_add`, missing nodes are never created; a
    path that does not exist yet is reported as 0, matching the value the
    node would hold right before its first increment.
    """
    lock = metadata.get(COUNTER_LOCK_KEY)
    if lock is None:
        lock = threading.Lock()
        metadata[COUNTER_LOCK_KEY] = lock

    with lock:
        tree = metadata.get(COUNTER_TREE_KEY)
        if tree is None:
            return 0

        node: dict[str, Any] | None = tree
        for part in key:
            assert isinstance(part, str)
            node = node["children"].get(part)
            if node is None:
                return 0
        return node["value"]


def global_counter_get_all(metadata: dict[str, Any], key: list[str]) -> dict[str, Any] | None:
    """Return a deep copy of the subtree rooted at ``key``, or ``None``.

    Walks the counter tree along ``key`` without creating missing nodes and
    returns a deep copy of the node found there (``{"value": ..., "children":
    ...}``), so callers can inspect it without racing concurrent updates.
    Returns ``None`` when the tree or any part of ``key`` does not exist.
    An empty ``key`` returns a copy of the whole tree.
    """
    lock = metadata.get(COUNTER_LOCK_KEY)
    if lock is None:
        lock = threading.Lock()
        metadata[COUNTER_LOCK_KEY] = lock

    with lock:
        tree = metadata.get(COUNTER_TREE_KEY)
        if tree is None:
            return None

        node: dict[str, Any] | None = tree
        for part in key:
            assert isinstance(part, str)
            node = node["children"].get(part)
            if node is None:
                return None
        return copy.deepcopy(node)
