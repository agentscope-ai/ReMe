"""steps — registers every BaseStep subclass at import time.

Each submodule's ``@R.register`` decorators only fire when the module
is imported. Auto-importing them here means any config that names a
step backend (e.g. ``graph_traverse_step``, ``write``, ``digester``)
will find it in the registry without the caller having to remember
which submodule it lives in.

File-I/O is split by blast radius. The ``crud`` package covers both
opaque-byte ops (list / stat / move / delete / upload / download) and
whole-file text ops (read / write / append / edit) — they share the
same path-resolution helpers, so they live in one package.
``frontmatter`` is the one sliced surface that earns its own RUD
package (YAML is structured data — surgical key edits cannot be safely
emulated with string-substitution on the body). For mid-file body
edits, use ``edit`` (exact string replacement) or do a read + write
round-trip.

* ``common``        — search / health_check / help / reindex / version / graph_traverse
* ``crud``          — list / stat / move / delete / upload / download / read / write / append / edit
* ``frontmatter``   — markdown frontmatter slice RUD (frontmatter_read_step / update / delete)
* ``daily``         — workspace genesis / list / day-index reindex
* ``jobs``          — synchronizer / digester (LLM-driven orchestrators)
"""

from . import common  # noqa: F401  -- registers common steps (search, version, graph_traverse, ...)
from . import crud  # noqa: F401  -- registers list/stat/upload/download/move/delete/read/write/append/edit
from . import frontmatter  # noqa: F401  -- registers frontmatter_read_step/update/delete
from . import daily  # noqa: F401  -- registers daily_resolve_step / daily_list_step / daily_reindex_step
from . import jobs  # noqa: F401  -- registers synchronizer / digester
from .base_step import BaseStep

__all__ = [
    "background",
    "common",
    "crud",
    "frontmatter",
    "daily",
    "jobs",
    "BaseStep",
]
