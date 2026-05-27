"""steps — registers every BaseStep subclass at import time.

Each submodule's ``@R.register`` decorators only fire when the module
is imported. Auto-importing them here means any config that names a
step backend (e.g. ``graph_traverse_step``, ``write``, ``digester``)
will find it in the registry without the caller having to remember
which submodule it lives in.

File-I/O is split by blast radius. The ``crud`` package covers
single-resource ops within the vault — both opaque-byte ops (list /
stat / move / delete) and whole-file text ops (read / write / append
/ edit), which share the same path-resolution helpers. The
``transfer`` package handles cross-domain bridges (vault ↔ local fs:
upload / download / upload_resource). ``frontmatter`` is the one
sliced surface that earns its own RUD package (YAML is structured
data — surgical key edits cannot be safely emulated with
string-substitution on the body). For mid-file body edits, use
``edit`` (exact string replacement) or do a read + write round-trip.

* ``common``        — health_check / help / version / traverse
* ``crud``          — list / stat / move / delete / read / write / append / edit
* ``transfer``      — upload / download / upload_resource (vault ↔ local fs)
* ``index``         — search / reindex / update_catalog / update_index
* ``frontmatter``   — markdown frontmatter slice RUD (frontmatter_read_step / update / delete)
* ``daily``         — note genesis / list / day-index reindex
* ``jobs``          — synchronizer / digester (LLM-driven orchestrators)
"""

from . import common  # noqa: F401  -- registers common steps (health_check, help, version, traverse, ...)
from . import crud  # noqa: F401  -- registers list/stat/move/delete/read/write/append/edit
from . import transfer  # noqa: F401  -- registers upload/download/upload_resource
from . import frontmatter  # noqa: F401  -- registers frontmatter_read_step/update/delete
from . import (
    daily,
)  # noqa: F401  -- registers daily_read_step / daily_write_step / daily_list_step / daily_reindex_step
from . import background  # noqa: F401
from . import index  # noqa: F401  -- registers update_catalog_step / update_index_step

# from . import jobs  # noqa: F401  -- registers synchronizer / digester
from .base_step import BaseStep

__all__ = [
    "background",
    "common",
    "crud",
    "transfer",
    "index",
    "frontmatter",
    "daily",
    "BaseStep",
]
