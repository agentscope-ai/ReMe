"""Thread-safe monotonic counter tree utility for shared application state."""

import threading
from typing import Any

COUNTER_TREE_KEY = "_counter_tree"
COUNTER_LOCK_KEY = "_counter_tree_lock"


def global_counter_next(metadata: dict[str, Any], key: list[str]) -> int:
    """Return the next monotonic value for ``key``, starting at 1.

    Walks the counter tree stored in ``metadata`` along ``key``, creating
    missing nodes on the way, then increments and returns the target node's
    counter. An empty ``key`` increments the root node, which serves as a
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
        res = node["value"] + 1
        node["value"] = res
    return res
