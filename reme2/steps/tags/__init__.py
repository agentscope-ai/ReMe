"""Tags steps — frontmatter-tag enumeration and per-tag statistics.

Two Steps:

    tags:list  — distinct tags + document counts across the vault.
    tags:stat  — count + file list for one specific tag.

Both consume what ``file_store`` already has in memory: path
discovery from ``file_store.file_chunks``, tag lookup from a single
batched ``file_graph.get_nodes(paths)`` call. No filesystem walk —
the cost is one dict iteration plus one graph round-trip.
"""

from . import list  # noqa: F401  -- @R.register("tags:list")
from . import stat  # noqa: F401  -- @R.register("tags:stat")
